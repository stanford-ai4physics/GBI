import pandas as pd
import numpy as np
from src.data_prep.utils import get_dijetmass_pxyz

TeV = 1000.0

def process_bkgs(input_path):
    """
    Will reprocess the bkgs such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=0)
    """

    data_all_df = pd.read_hdf(input_path)

    # bkgs only
    if "label" in data_all_df.columns:
        data_all_df = data_all_df.query("label == 0")

    # get jet p4 info to calculate dijet mjj
    jet1_p4 = data_all_df[["pxj1", "pyj1", "pzj1", "mj1"]].values
    jet2_p4 = data_all_df[["pxj2", "pyj2", "pzj2", "mj2"]].values
    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_pxyz(jets) / TeV

    # get other features
    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.array(data_all_df[['mj1', 'mj2']]) / TeV
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1 = data_all_df["tau2j1"].values / ( 1e-5 + data_all_df["tau1j1"].values )
    tau21j2 = data_all_df["tau2j2"].values / ( 1e-5 + data_all_df["tau1j2"].values )
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack([mjj, mjmin, mjmax-mjmin, tau21min, tau21max, np.zeros(len(mj1mj2))], axis=1)

    return output

