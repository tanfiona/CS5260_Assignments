conda create --name pytorch python=3.9
conda activate pytorch
conda install jupyter
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install colossalai