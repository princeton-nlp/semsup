# Semantic Supervision: Enabling Generalization over Output Spaces

[**Website**](https://sites.google.com/princeton.edu/semantic-supervision/) |
[**Paper**](https://arxiv.org/abs/2202.13100)

![semsup](https://user-images.githubusercontent.com/8196527/200193827-9f1fb40b-35f3-4e6b-a685-0005319b6bbc.gif)

## Setup
First clone the repository and install dependencies:
```
git clone https://github.com/princeton-nlp/semsup.git
pip install -e semsup
```

Inside the `semsup` folder, download class descriptions and make the cache folder:
```
cd semsup
bash download.sh
mkdir data_cache
```

Experiments in our paper were run using `python 3.9.7` and `CUDA toolkit 11.1`.

## How to run the code
Scripts and config files to train and evaluate models on AWA2, CIFAR and 20Newsgroups are found in the folders `run_{awa, cifar, ng}`. Training commands are found in the bash scripts `run_{dataset}_{model}.sh` in each folder. Each config file has a `scen{1,2,3}` in the file name to indicate which scenario the config is for (see the paper for details on the scenarios). Run each script inside the respective folder. For example:
```
cd run_cifar
bash run_cifar_semsup.sh
```

Dataset downloading to the `data_cache` directory is automatically handled. Test commands are provided in the `test_{awa, cifar, ng}.sh` scripts. Note that you will need to change the `--checkpoints` argument to point to where the model weights are stored.

### RCV1
Obtaining RCV1 data and running RCV1 code require additional instructions that can be found [here](run_rcv1/RCV1_README.md).

## Notes
### Building on this code
We use `pytorch-lightning` (except for RCV1). The `SemSupDataModule` class in `semsup/data/core.py` inherits from `pl.LightningDataModule` and handles the tokenization, caching, and sampling of class descriptions. All data modules in `semsup/data/{awa, cifar, newsgroups}` inherit from `SemSupDataModule`.

The `SemSupModel` in `semsup/model/core.py` inherits from `pl.LightningModule` and constructs the output matrix. All SemSup models inherit from this class. Together, the two base `SemSup` classes allow for easy conversion of any supervised problem into a `SemSup` approach.

The class descriptions `.labels` files are a list of jsons with `"text"` and `"label"` keys. The label be one of the labels provided in the `train_classes` and `val_classes` variables in `SemSupDataArgs`.

### Running in Colab
(For AWA, CIFAR, Newsgroups). To run the code in Google Colab, make sure to run the following cell first to setup your environment.
```
!git clone https://github.com/princeton-nlp/semsup.git
%cd semsup/
!pip uninstall --yes torchtext torchaudio # not compatible with our version of torch
!pip install -e .
!bash download.sh
!mkdir data_cache
# RESTART RUNTIME TO UPDATE NEWLY INSTALLED MODULES
```
You should now be able to run the scripts (make sure to `cd` to the folder) for example:
```
%cd /content/semsup/run_ng
!bash unit_test_ng.sh
```
### License
MIT
