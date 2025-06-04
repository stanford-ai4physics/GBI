# import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle
from config.configs import SR_MAX, SR_MIN
from src.utils.utils import NumpyEncoder


def separate_SB_SR(data):
    innermask = (data[:, 0] > SR_MIN) & (data[:, 0] < SR_MAX)
    outermask = ~innermask
    return data[innermask], data[outermask]


def background_split(background, resample_seed=42):

    # shuffle data
    background = shuffle(background, random_state=resample_seed)

    # split bkg into SR and CR
    SR_bkg, CR_bkg = separate_SB_SR(background)

    print("SR bkg shape: ", SR_bkg.shape)
    print("CR bkg shape: ", CR_bkg.shape)

    return SR_bkg, CR_bkg


# def resample_split_test(signal_path, bkg_path, resample_seed = 42):

#     background = np.load(bkg_path)
#     signal = np.load(signal_path)

#     # shuffle data
#     background = shuffle(background, random_state=resample_seed)
#     signal = shuffle(signal, random_state=resample_seed)

#     # split bkg into SR and CR
#     SR_bkg, CR_bkg = separate_SB_SR(background)

#     SR_sig, CR_sig = separate_SB_SR(signal)
#     # for now we ignore signal in CR

#     SR_sig_injected = SR_sig[:50000]

#     # concatenate background and signal
#     SR_data_test = np.concatenate((SR_bkg, SR_sig_injected),axis=0)
#     SR_data_test = shuffle(SR_data_test, random_state=resample_seed)

#     print('SR test shape: ', SR_data_test.shape)
#     print('SR test num sig: ', (SR_data_test[:, -1]==1).sum())

#     return SR_data_test


def shuffle_trainval(input, output, resample_seed=42):

    # first load data

    SR_data_trainval_model_S = np.load(
        input["preprocessing"]["SR_data_trainval_model_S"].path
    )
    SR_data_trainval_model_B = np.load(
        input["preprocessing"]["SR_data_trainval_model_B"].path
    )
    with open(input["preprocessing"]["SR_mass_hist"].path, "r") as f:
        mass_hist = json.load(f)

    log_B_trainval = np.load(input["bkgprob"]["log_B_trainval"].path)

    # split data into train and val using the same random index, train-val split is 2:1
    np.random.seed(resample_seed)
    random_index_train = np.random.choice(
        SR_data_trainval_model_S.shape[0],
        int(SR_data_trainval_model_S.shape[0] * 2 / 3),
        replace=False,
    )
    random_index_val = np.setdiff1d(
        np.arange(SR_data_trainval_model_S.shape[0]), random_index_train
    )

    SR_data_train_model_S = SR_data_trainval_model_S[random_index_train]
    SR_data_val_model_S = SR_data_trainval_model_S[random_index_val]

    SR_data_train_model_B = SR_data_trainval_model_B[random_index_train]
    SR_data_val_model_B = SR_data_trainval_model_B[random_index_val]

    log_B_train = log_B_trainval[random_index_train]
    log_B_val = log_B_trainval[random_index_val]

    # save data
    np.save(
        output["preprocessing"]["data_train_SR_model_S"].path, SR_data_train_model_S
    )
    np.save(output["preprocessing"]["data_val_SR_model_S"].path, SR_data_val_model_S)
    np.save(
        output["preprocessing"]["data_train_SR_model_B"].path, SR_data_train_model_B
    )
    np.save(output["preprocessing"]["data_val_SR_model_B"].path, SR_data_val_model_B)
    with open(output["preprocessing"]["SR_mass_hist"].path, "w") as f:
        json.dump(mass_hist, f, cls=NumpyEncoder)

    np.save(output["bkgprob"]["log_B_train"].path, log_B_train)
    np.save(output["bkgprob"]["log_B_val"].path, log_B_val)
