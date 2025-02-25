This repository contains the code used in the paper ["Incorporating Physical Priors into Weakly-Supervised Anomaly Detection"](https://arxiv.org/abs/2405.08889) by Chi Lung Cheng, Gurpreet Singh, and Benjamin Nachman.

## Introduction

In this work, we propose a new machine-learning-based anomaly detection strategy called Prior-Assisted Weak Supervision (PAWS). This approach is designed to improve the sensitivity of weakly supervised anomaly detection methods by incorporating information from a class of signal models. PAWS matches the sensitivity of fully supervised methods when the true signal falls within the pre-specified class, without requiring the exact parameters to be known in advance. This method significantly extends the sensitivity of searches, even in scenarios with rare signals or many irrelevant features.

Our study demonstrates the effectiveness of PAWS on the Large Hadron Collider Olympics (LHCO) benchmark dataset, achieving a tenfold increase in sensitivity over previous methods. PAWS remains robust in the presence of noise, where classical methods fail. This approach has broad applicability and advances the frontier of sensitivity between model-agnostic and model-specific searches.

## Toy Demonstration

A demonstration illustrating the PAWS method using toy datasets and distributions without need for external packages is available as github gist here:

[View the Gist](https://gist.github.com/AlkaidCheng/6778da484fb49f19aca26d6d3d290db1)

## Installation

PAWS provides both API and CLI interfaces. The code requires python 3.8+ and the following libraries:

```python
numpy==1.26.2
matplotlib==3.8.2
pandas==2.1.3
awkard==2.6.2
vector==1.4.0
aliad==0.2.0
quickstats==0.7.0.8
tensorflow==2.15.0
```

The dependencies are also available in `requirements.txt` which can be installed via pip. Make sure you install tensorflow with gpu support if you want to train with GPUs.

```
pip install -r requirements.txt
```

To setup paws, the simplest way will be to source the setup script:

```
source setup.sh
```

Alternatively, you may install it via `setup.py`:
```
pip install git+https://github.com/hep-lbdl/PAWS.git
```

## Datasets

The data samples used in the paper can be downloaded through Zenodo:

- QCD background from official LHCO dataset: https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5
- Extra QCD background : https://zenodo.org/records/8370758/files/events_anomalydetection_qcd_extra_inneronly_features.h5
- Parametric W->X(qq)Y(qq) signal : https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5
- Parametric W->X(qq)Y(qqq) signal : https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qqq_parametric.h5

## Tutorials

### Command Line Interface

To checkout what commands are available in the paws CLI:
```
paws --help
```

The paws CLI provides a set of subcommands for data preparation, training and evaluation of the models used in the paper. To get help on how to use them:
```
paws <subcommand> --help
```

Below are some example commands to reproduce results in the paper. More detailed tutorials are available as jupyter notebooks in the [tutorials](tutorials) directory.

#### Data Preparation

- Download data samples from Zenodo:
```
paws download_data -d datasets
```
- Create dedicated datasets in tfrecord format:
```
paws create_dedicated_datasets -s QCD,extra_QCD,W_qq,W_qqq -d datasets
```

- Create parameterised dataests in tfrecord format:
```
# two-prong dataset
paws create_param_datasets -s QCD,extra_QCD,W_qq -d datasets
# three-prong dataset
paws create_param_datasets -s QCD,extra_QCD,W_qqq -d datasets
```

#### Model Training

- Train dedicated supervised models:
```
# usage: paws train_dedicated_supervised \[--options\]
# e.g. 2-prong + 3-prong signals, (mX, mY) = 300, 300 GeV
paws -d datasets -o outputs --mass-point 300:300 --decay-modes qq,qqq --variables 3,5,6 --split-index 0 --version v1
```

- Train parameterised supervised models:
```
# usage: paws train_param_supervised \[--options\]
# e.g. 2-prong signal
paws train_param_supervised -d datasets -o outputs --decay-modes qq --variables 3,5,6 --split-index 0 --version v1
# e.g. 3-prong signal
paws train_param_supervised -d datasets -o outputs --decay-modes qq --variables 3,5,6 --split-index 0 --version v1
```

- Train ideal weakly models:
```
# usage: paws train_ideal_weakly \[--options\]
# e.g. 2-prong + 3-prong signals, (mX, mY) = 300, 300 GeV, signal fraction mu = 0.01, decay branching ratio alpha = 0.5
paws train_ideal_weakly -d datasets -o outputs --mass-point 300:300 --decay-modes qq,qqq --variables 3,5,6 --mu 0.01 --alpha 0.5 --split-index 0 --version v1
```

- Train semi-weakly (PAWS) models:
```
# usage: paws train_semi_weakly \[--options\]
# e.g. 2-prong + 3-prong signals, (mX, mY) = 300, 300 GeV, signal fraction mu = 0.01, decay branching ratio alpha = 0.5
paws train_semi_weakly -d datasets -o outputs --mass-point 300:300 --decay-modes qq,qqq --variables 3,5,6 --mu 0.01 --alpha 0.5 --split-index 0 --version v1 --fs-version v1
```

#### Evaluation

- Compute metric landscapes:
```
# e.g. 2-prong + 3-prong signals, (mX, mY) = 300, 300 GeV, signal fraction mu = 0.005
paws compute_semi_weakly_landscape -m 300:300 --mu 0.005 --decay-modes qq,qqq --variables 3,5,6 -d datasets -o outputs
```

- Gather model results:
```
paws gather_model_results --model-type dedicated_supervised --variables "3,5,6" -o outputs
paws gather_model_results --model-type ideal_weakly --variables "3,5,6" -o outputs
paws gather_model_results --model-type semi_weakly --variables "3,5,6" -o outputs
```

### Animations

<p align="center">
  <img src="assets/gifs/demo1.gif" alt="Animation 1" width="45%">
  <img src="assets/gifs/demo2.gif" alt="Animation 2" width="45%">
</p>

The code for the illustrative animations can be found at [here](assets/landscapes.ipynb). Also found in the notebook is the interpolated plot used in the paper.

### Jupyter Notebooks

More detailed tutorials are available in the form of jupyter notebooks which cover both CLI and API usage. They can be found in the [tutorials](tutorials) directory. 

- [Tutorial-00 Toy Demonstration](tutorials/T00-ToyDemonstration.ipynb)

- [Tutorial-01 Data Preparation](tutorials/T01-DataPreparation.ipynb)

- [Tutorial-02 Data Loading](tutorials/T02-DataLoading.ipynb)

- [Tutorial-03 Model Loading](tutorials/T03-ModelLoading.ipynb)

- [Tutorial-04 Model Training](tutorials/T04-ModelTraining.ipynb)

- [Tutorial-05 Gather Results](tutorials/T05-GatherResults.ipynb)

- [Tutorial-06 Plotting](tutorials/T06-Plotting.ipynb)

- [Tutorial-07 Metric Landscape](tutorials/T07-MetricLandscape.ipynb)
  
## Citation
```python
@article{Cheng:2024yig,
    author = "Cheng, Chi Lung and Singh, Gurpreet and Nachman, Benjamin",
    title = "{Incorporating Physical Priors into Weakly-Supervised Anomaly Detection}",
    eprint = "2405.08889",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "5",
    year = "2024"
}
```