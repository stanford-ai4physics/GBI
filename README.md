This repository contains the code used in the paper ["Generator Based Inference (GBI)"](https://arxiv.org/abs/2405.08889) by Chi Lung Cheng, Ranit Das, Runze Li, Radha Mastandrea, Vinicius Mikuni, Benjamin Nachman, David Shih and Gup Singh.

## Introduction

This repository contains the code and resources for the paper "Generator Based Inference (GBI)". The work introduces GBI as a general framework for integrating machine learning with statistical inference based on generative models. While Simulation Based Inference (SBI) is a well-known instance where the generator is a physics-based simulator, GBI extends this concept to include scenarios where the generator itself is learned from data. This is particularly relevant for analyses where background models are predominantly data-driven.

The paper specifically demonstrates the power of GBI in the context of resonant anomaly detection. It explores methods where the background generator is learned from sideband data, and then utilized for machine learning-based parameter estimation of potential signals. Two key approaches, R-ANODE (which learns signal and background from data) and an extension of PAWS (a data-simulation hybrid), are investigated. These methods transform the outputs of anomaly detection into directly interpretable results, such as confidence intervals on signal strength and physical parameters, achieving state-of-the-art sensitivity on the LHCO community benchmark dataset.

## Installation

GBI provides both API and CLI interfaces. The code requires python 3.8+ and the following libraries:

```python
numpy==1.26.2
matplotlib==3.8.2
pandas==2.1.3
awkard==2.6.2
vector==1.4.0
aliad==0.2.0
quickstats==0.8.3.5.11
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
pip install git+https://github.com/hep-lbdl/paws-sbi.git
```

## Datasets

The data samples used in the paper can be downloaded through Zenodo:

- Simulated QCD background from official LHCO dataset: https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5
- Extra simulated QCD background : https://zenodo.org/records/8370758/files/events_anomalydetection_qcd_extra_inneronly_features.h5
- Parametric W->X(qq)Y(qq) signal : https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5
- Parametric W->X(qq)Y(qqq) signal : https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qqq_parametric.h5
- Extended parametric W->X(qq)Y(qq) signal : https://zenodo.org/records/15384386/files/events_anomalydetection_extended_Z_XY_qq_parametric.h5
- Generative QCD background : https://zenodo.org/records/15384501/files/events_anomalydetection_generative_features.h5

## Tutorials

### Command Line Interface

#### Data Preparation

#### Model Training

#### Evaluation

### Jupyter Notebooks

## Citation
