### Introduction
We experimented with the [Colossal AI](https://github.com/hpcaitech/ColossalAI) library to train LeNet5 on MNIST.

The hyperparameters and architectures we played with for this assignment is as follows:
* Optimizers: ['sgd','adamw']
* Learning rate schedulers: ['lambda','multistep']

Please check `report.pdf` for a summary of our findings.

### Requirements
Set up a new conda environment based off `environment.yml` or using `requirements.txt`.
Alternatively, run `setup.sh` to install the environment and dependencies.

### Codes
To perform the LR Range Test, use the Jupyter Notebook `Colossalai LR Range Tests.ipynb`.

To run the main train script with the experiments, use `run.py`. Run in command line: `python3 run.py`.

To visualize tensorboard and to obtain plots, refer to `Visualize with Tensorboard.ipynb`.

### Contact
Fiona Tan, tan.f[at]u.nus.edu 
