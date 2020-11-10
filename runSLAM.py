import numpy as np
from typing import List, Optional
from matplotlib import pyplot as plt
from EKFSLAM import EKFSLAM
from tqdm import tqdm
from scipy.stats import chi2


def run_slam(Q, R, JCBBalphas, eta_pred_init, P_pred_init,
             odometry, z, poseGT, N, alpha, do_asso=True, doAssoPlot=False):
    K = len(z)
    slam = EKFSLAM(Q, R, do_asso=do_asso, alphas=JCBBalphas)

    axAsso = None

    # allocate
    eta_pred: List[Optional[np.ndarray]] = [None] * K
    P_pred: List[Optional[np.ndarray]] = [None] * K
    eta_hat: List[Optional[np.ndarray]] = [None] * K
    P_hat: List[Optional[np.ndarray]] = [None] * K
    a: List[Optional[np.ndarray]] = [None] * K
    NIS = np.zeros(K)
    NISnorm = np.zeros(K)
    CI = np.zeros((K, 2))
    CInorm = np.zeros((K, 2))
    NEES = np.zeros((K, 3))

    eta_pred[0] = eta_pred_init
    P_pred[0] = P_pred_init
    # %% Run simulation

    for k, z_k in tqdm(enumerate(z[:N])):
        eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(
            eta_pred[k], P_pred[k], z_k)  # TODO update
        if k < K - 1:
            # TODO predict
            eta_pred[k + 1], P_pred[k +
                                    1] = slam.predict(eta_hat[k], P_hat[k], odometry[k])

        # assert (
        #     eta_hat[k].shape[0] == P_hat[k].shape[0]
        # ), "dimensions of mean and covariance do not match"

        num_asso = np.count_nonzero(a[k] > -1)

        CI[k] = chi2.interval(alpha, 2 * num_asso)

        if num_asso > 0:
            NISnorm[k] = NIS[k] / (2 * num_asso)
            CInorm[k] = CI[k] / (2 * num_asso)
        else:
            NISnorm[k] = 1
            CInorm[k].fill(1)

        NEES[k] = EKFSLAM.NEESes(eta_hat[k][:3], P_hat[k][:3, :3], poseGT[k])

    return eta_pred, P_pred, eta_hat, P_hat, a, NIS, NISnorm, CI, CInorm, NEES
