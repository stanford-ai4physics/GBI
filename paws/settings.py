import os

import numpy as np

from quickstats.core import GeneralEnum, DescriptiveEnum

class KeyDescriptiveEnum(DescriptiveEnum):
    def __new__(cls, value: int, description: str = "", key: str = ""):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.key = key
        return obj

class FeatureLevel(KeyDescriptiveEnum):
    HIGH_LEVEL = (0, "High level jet features", "high_level")
    LOW_LEVEL  = (1, "Low level particle features + high level jet features", "low_level")
HIGH_LEVEL = FeatureLevel.HIGH_LEVEL
LOW_LEVEL  = FeatureLevel.LOW_LEVEL

class DecayMode(KeyDescriptiveEnum):
    __aliases__ = {
        "qq" : "two_prong",
        "qqq": "three_prong"
    }
    TWO_PRONG   = (0, "Two-prong decay W'->X(qq)Y(qq)", "qq")
    THREE_PRONG = (1, "Three-prong decay W'->X(qq)Y(qqq)", "qqq")
TWO_PRONG   = DecayMode.TWO_PRONG
THREE_PRONG = DecayMode.THREE_PRONG

class ModelType(KeyDescriptiveEnum):
    DEDICATED_SUPERVISED = (0, "Supervised training at a dedicated mass point", "dedicated_supervised")
    PARAM_SUPERVISED     = (1, "Supervised training with parametric masses", "param_supervised")
    IDEAL_WEAKLY         = (2, "Ideal weakly supervised training", "ideal_weakly")
    SEMI_WEAKLY          = (3, "Semi-weakly supervised training", "semi_weakly")
    PRIOR_RATIO          = (4, "Prior ratio for calibrating supervised model output", "prior_ratio")
DEDICATED_SUPERVISED = ModelType.DEDICATED_SUPERVISED
PARAM_SUPERVISED = ModelType.PARAM_SUPERVISED
IDEAL_WEAKLY = ModelType.IDEAL_WEAKLY
SEMI_WEAKLY = ModelType.SEMI_WEAKLY
PRIOR_RATIO = ModelType.PRIOR_RATIO

class Sample(DescriptiveEnum):
    
    QCD       = (0, "QCD dijet background", "QCD", 0, -1, None, True)
    EXTRA_QCD = (1, "Extended QCD dijet background", "extra_QCD", 0, -1, None, True)
    W_QQ      = (2, "W'->X(qq)Y(qq) signal", "W_qq", 1, 1, TWO_PRONG, True)
    W_QQQ     = (3, "W'->X(qq)Y(qqq) signal", "W_qqq", 1, 1, THREE_PRONG, True)    

    QCD_GEN   = (4, 'QCD dijet background (generative)', 'QCD_gen', 0, 0, None, False)
    MIXED     = (5, 'Mixed sample', 'mixed', None, None, None, False)

    EXTRA_W_QQ = (6, "Extended W'->X(qq)Y(qq) signal", "extra_W_qq", 1, 1, TWO_PRONG, True)
    EXTRA_W_QQQ = (7, "Extended W'->X(qq)Y(qqq) signal", "extra_W_qqq", 1, 1, THREE_PRONG, True)
    
    def __new__(cls, value: int, description:str, key:str,
                supervised_label: int, weakly_label: int,
                decay_mode:DecayMode, default_sample: bool):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.key = key
        obj.supervised_label = supervised_label
        obj.weakly_label = weakly_label
        obj.decay_mode = decay_mode
        obj.default_sample = default_sample
        return obj
        
QCD = Sample.QCD
EXTRA_QCD = Sample.EXTRA_QCD
W_QQ = Sample.W_QQ
W_QQQ = Sample.W_QQQ
QCD_GEN = Sample.QCD_GEN

# dataset setups
#MASS_RANGE = (50, 600)
MASS_RANGE = (50, 600)
MASS_INTERVAL = 50
NUM_JETS = 2
NUM_SHARDS = 100

# train setups
SPLIT_FRACTIONS = "50:25:25"
SAMPLING_METHOD = "sampled"
BASE_SEED = 2023
NUM_TRIALS = 10
WEIGHT_CLIPPING = True
RETRAIN = False

SEMI_WEAKLY_PARAMETERS = ['m1', 'm2', 'mu', 'alpha']
SIGMOID_ACTIVATION = True
MASS_SCALE = 1 / 100
INIT_MU = np.log(np.exp(-4) / (1 - np.exp(-4))) if SIGMOID_ACTIVATION else -4
INIT_ALPHA = np.log(np.exp(-1) / (1 - np.exp(-1))) if SIGMOID_ACTIVATION else np.exp(-1)
INIT_KAPPA = 1.

# List of signal fractions and decay branching ratios used in this study
# Note the numbers were a bit off from the paper values because paper used S / B instead of S / (B + S) by mistake when defining the datasets so those values have to be shifted a little to correct for it (little to no effect for small mu). This mistake has now been fixed in the code.
MU_LIST = np.logspace(-3.5, -1.3, 10)
ALPHA_LIST = np.array([0.5])

MLP_LAYERS = [(256, 'relu'),
              (128, 'relu'),
              (64, 'relu'),
              (1, 'sigmoid')]

PRIOR_RATIO_NET_LAYERS = [(64, 'relu', 0.0001, 0.1),
                          (32, 'relu', 0.0001, 0.1),
                          (1, None, None, None)]

# names of trainable features as in dataset (ordering is important)
TRAIN_FEATURES = {
    HIGH_LEVEL : ['jet_features'],
    LOW_LEVEL  : ['part_coords', 'part_features', 'part_masks', 'jet_features']
}

# name of physical parameters as in dataset
PARAM_FEATURE = 'param_masses'

# sample naming formats
SIG_SAMPLE = 'W_{decay_mode}'
BKG_SAMPLE = 'QCD_qq'
EXT_BKG_SAMPLE = 'extra_QCD_qq'
PARAM_SIG_SAMPLE = 'W_{decay_mode}_{m1}_{m2}'
PARAM_BKG_SAMPLE = 'QCD_qq_{m1}_{m2}'
PARAM_EXT_BKG_SAMPLE = 'extra_QCD_qq_{m1}_{m2}'

DEFAULT_FEATURE_LEVEL = "high_level"
DEFAULT_DECAY_MODE = "qq,qqq"
DEFAULT_DATADIR = "datasets"
DEFAULT_OUTDIR = "outputs"

CHECKPOINT_DIR_FMTS = {
    DEDICATED_SUPERVISED : ("dedicated_supervised/"
                            "{feature_level}/{decay_mode}/{m1}_{m2}/"
                            "SR_var_{variables}_noise_{noise_dim}_{version}/"
                            "split_{split_index}"),
    PARAM_SUPERVISED     : ("param_supervised/"
                            "{feature_level}/{decay_mode}/"
                            "SR_var_{variables}_noise_{noise_dim}_{version}/"
                            "split_{split_index}"),
    IDEAL_WEAKLY         : ("ideal_weakly/"
                            "{feature_level}/{decay_mode}/{m1}_{m2}/"
                            "SR_var_{variables}_noise_{noise_dim}_{version}/{mu_alpha}/"
                            "split_{split_index}/trial_{trial}"),
    SEMI_WEAKLY          : ("semi_weakly/"
                            "{feature_level}/{decay_mode}/{m1}_{m2}/"
                            "SR_var_{variables}_noise_{noise_dim}_{version}/{mu_alpha}/"
                            "split_{split_index}/trial_{trial}")
}

FEILDS_REGEX = {
    "decay_mode": r"(?P<decay_mode>\w+)",
    "feature_level": r"(?P<feature_level>\w+)",
    "m1": r"(?P<m1>\d+)",
    "m2": r"(?P<m2>\d+)",
    "variables": r'(?P<variables>all|\d[0-9_]*)',
    "version": r"(?P<version>\w+)",
    "split_index": r"(?P<split_index>\d+)",
    "mu_alpha": r'(mu_(?P<mu>\dp\d+))(_alpha_(?P<alpha>\dp\d+))?',
    "noise_dim": r"(?P<noise_dim>\d+)",
    "trial": r"(?P<trial>\d+)"
}

kSampleList = ["QCD", "extra_QCD", "W_qq", "W_qqq"]
SampleURLs = {
    QCD: "https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5",
    EXTRA_QCD: "https://zenodo.org/records/8370758/files/events_anomalydetection_qcd_extra_inneronly_features.h5",
    W_QQ: "https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5",
    W_QQQ: "https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qqq_parametric.h5",
    QCD_GEN: "events_anomalydetection_cathode_features.h5"
}