from json.tool import main
import os
from pathlib import Path
import math
import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits


def run(optimizer_method, scheduler_method, learning_rate):
    
    logger = get_dist_logger()
    d_name = f'{optimizer_method}_{scheduler_method}_{learning_rate}'
    logger.log_to_file(f'./tb_logs2/{d_name}')
    logger.info(f'>>>>>>{optimizer_method}_{scheduler_method}_{learning_rate}')

    # build 
    model = LeNet5(n_classes=10)

    # build dataloaders
    train_dataset = MNIST(
        root=Path('./tmp/'),
        download=True,
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor()])
    )

    test_dataset = MNIST(
        root=Path('./tmp/'),
        train=False,
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor()])
    )

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                      add_sampler=False,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if optimizer_method.lower()=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif optimizer_method.lower()=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    else:
        raise NotImplementedError

    # # lr_scheduler
    # if scheduler_method.lower()=='lambda':
    #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer)
    # elif scheduler_method.lower()=='multistep':
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, 
    #         milestones=[i*len(train_dataloader) for i in range(gpc.config.NUM_EPOCHS)], 
    #         gamma=0.1)
    # elif scheduler_method.lower()=='onecycle':
    #     lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer, max_lr=learning_rate/10,
    #         steps_per_epoch=len(train_dataloader), 
    #         epochs=gpc.config.NUM_EPOCHS
    #     )
    # else:
    #     raise NotImplementedError

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
    
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        # hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        hooks.TensorboardHook(log_dir=f'./tb_logs2/{d_name}', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    # Combinations, LR from LR Range Test
    combinations = [
        ('sgd','',0.025),
        ('sgd','',0.0025),
        ('sgd','',0.25),
        ('adamw','',0.001),
        ('adamw','',0.0001),
        ('adamw','',0.01),
    ]

    # Single launch
    config = {'BATCH_SIZE':128,'NUM_EPOCHS':30}
    colossalai.launch(config=config,rank=0,world_size=1,host='127.0.0.1',port=1234)

    # Loop
    for (optimizer_method, scheduler_method, learning_rate) in combinations:
        run(optimizer_method, scheduler_method, learning_rate)