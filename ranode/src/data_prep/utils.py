import numpy as np
from sklearn.utils import shuffle


def get_dijetmass_ptetaphi(jets):
    jet_e = np.sqrt(
        jets[:, 0, 3] ** 2 + jets[:, 0, 0] ** 2 * np.cosh(jets[:, 0, 1]) ** 2
    )
    jet_e += np.sqrt(
        jets[:, 1, 3] ** 2 + jets[:, 1, 0] ** 2 * np.cosh(jets[:, 1, 1]) ** 2
    )
    jet_px = jets[:, 0, 0] * np.cos(jets[:, 0, 2]) + jets[:, 1, 0] * np.cos(
        jets[:, 1, 2]
    )
    jet_py = jets[:, 0, 0] * np.sin(jets[:, 0, 2]) + jets[:, 1, 0] * np.sin(
        jets[:, 1, 2]
    )
    jet_pz = jets[:, 0, 0] * np.sinh(jets[:, 0, 1]) + jets[:, 1, 0] * np.sinh(
        jets[:, 1, 1]
    )
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj


def get_dijetmass_pxyz(jets):
    jet_e = np.sqrt(
        jets[:, 0, 3] ** 2
        + jets[:, 0, 0] ** 2
        + jets[:, 0, 1] ** 2
        + jets[:, 0, 2] ** 2
    )
    jet_e += np.sqrt(
        jets[:, 1, 3] ** 2
        + jets[:, 1, 0] ** 2
        + jets[:, 1, 1] ** 2
        + jets[:, 1, 2] ** 2
    )
    jet_px = jets[:, 0, 0] + jets[:, 1, 0]
    jet_py = jets[:, 0, 1] + jets[:, 1, 1]
    jet_pz = jets[:, 0, 2] + jets[:, 1, 2]
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj


def standardize(x, mean, std):
    return (x - mean) / std


def logit_transform(x, min_vals, max_vals):
    with np.errstate(divide="ignore", invalid="ignore"):
        x_norm = (x - min_vals) / (max_vals - min_vals)
        logit = np.log(x_norm / (1 - x_norm))
    domain_mask = ~(np.isnan(logit).any(axis=1) | np.isinf(logit).any(axis=1))
    return logit, domain_mask


def inverse_logit_transform(x, min_vals, max_vals):
    x_norm = 1 / (1 + np.exp(-x))
    return x_norm * (max_vals - min_vals) + min_vals


def inverse_standardize(x, mean, std):
    return x * std + mean


def preprocess_params_fit(data):
    preprocessing_params = {}
    preprocessing_params["min"] = np.min(data[:, 1:-1], axis=0)
    preprocessing_params["max"] = np.max(data[:, 1:-1], axis=0)

    preprocessed_data_x, mask = logit_transform(
        data[:, 1:-1], preprocessing_params["min"], preprocessing_params["max"]
    )
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[
        mask
    ]

    preprocessing_params["mean"] = np.mean(preprocessed_data[:, 1:-1], axis=0)
    preprocessing_params["std"] = np.std(preprocessed_data[:, 1:-1], axis=0)

    return preprocessing_params


def preprocess_params_transform(data, params):
    preprocessed_data_x, mask = logit_transform(
        data[:, 1:-1], params["min"], params["max"]
    )
    preprocessed_data = np.hstack([data[:, 0:1], preprocessed_data_x, data[:, -1:]])[
        mask
    ]
    preprocessed_data[:, 1:-1] = standardize(
        preprocessed_data[:, 1:-1], params["mean"], params["std"]
    )
    return preprocessed_data


def inverse_transform(data, params):
    inverse_data = inverse_standardize(
        data[:, 1:-1], np.array(params["mean"]), np.array(params["std"])
    )
    inverse_data = inverse_logit_transform(
        inverse_data, np.array(params["min"]), np.array(params["max"])
    )
    inverse_data = np.hstack([data[:, 0:1], inverse_data, data[:, -1:]])

    return inverse_data


def fold_splitting(
    data,
    n_folds=5,
    random_seed=42,
    test_fold=0,
):
    np.random.seed(random_seed)

    data = shuffle(data, random_state=random_seed)

    # split into fold_split_num folds, test fold is fold_split_seed-th fold
    data_folds = {}
    for fold in range(n_folds):
        data_folds[fold] = data[
            fold * int(len(data) / n_folds) : (fold + 1) * int(len(data) / n_folds)
        ]

    # get the signal trainval and test set index
    sig_test_index = test_fold
    sig_trainval_index_list = [i for i in range(n_folds) if i != sig_test_index]

    data_test = data_folds[sig_test_index]
    data_trainval = np.concatenate([data_folds[i] for i in sig_trainval_index_list])

    return data_trainval, data_test
