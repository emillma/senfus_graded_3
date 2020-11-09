# %% Imports
from debugger import mes_diff
from plotting import ellipse
from EKFSLAM import EKFSLAM
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils

from tqdm import tqdm
from plott_setup import setup_plot
from plotting import *
setup_plot()
# %% plot config check and style setup

# to see your plot config


# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")


z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T

K = len(z)
M = len(landmarks)

# %% Initilize
Q = np.diag([0.1, 0.1, 0.001])  # TODO Best
R = np.diag([0.0025, 0.0004])  # TODO Best
assert 0
doAsso = True

JCBBalphas = np.array(
    [10e-4, 10e-6]  # TODO,
)   # first is for joint compatibility, second is individual
# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.


slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)


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
NEESes = np.zeros((K, 3))

# For consistency testing
alpha = 0.05

# init
eta_pred[0] = poseGT[0]  # we start at the correct position for reference
P_pred[0] = np.zeros((3, 3))  # we also say that we are 100% sure about that

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = True
if doAssoPlot:
    figAsso, axAsso = plt.subplots(num=1, clear=True)

# %% Run simulation
N = 1000

print("starting sim (" + str(N) + " iterations)")

for k, z_k in tqdm(enumerate(z[:N])):
    eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(
        eta_pred[k], P_pred[k], z_k)  # TODO update
    if k < K - 1:
        # TODO predict
        eta_pred[k + 1], P_pred[k +
                                1] = slam.predict(eta_hat[k], P_hat[k], odometry[k])

    assert (
        eta_hat[k].shape[0] == P_hat[k].shape[0]
    ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI[k] = chi2.interval(alpha, 2 * num_asso)

    if num_asso > 0:
        NISnorm[k] = NIS[k] / (2 * num_asso)
        CInorm[k] = CI[k] / (2 * num_asso)
    else:
        NISnorm[k] = 1
        CInorm[k].fill(1)

    # NEESes[k] =  # TODO, use provided function slam.NEESes

    if doAssoPlot and k > 0:
        axAsso.clear()
        axAsso.grid()
        zpred = slam.h(eta_pred[k]).reshape(-1, 2)
        axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
        axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
        xcoords = np.block(
            [[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
        ycoords = np.block(
            [[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
        for x, y in zip(xcoords, ycoords):
            axAsso.plot(x, y, lw=3, c="r")
        axAsso.legend()
        axAsso.set_title(
            f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
        plt.draw()
        plt.pause(0.001)


print("sim complete")

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
a = play_movie(pose_est, poseGT, lmk_est, landmarks, P_hat, N)
plot_trajectory(pose_est, poseGT, P_hat, N)
plot_NIS(NISnorm, CInorm, N)
plt.show()
