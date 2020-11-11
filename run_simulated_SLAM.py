# %% Imports
from scipy.io import loadmat
import numpy as np

import matplotlib.pyplot as plt

from plott_setup import setup_plot
from plotting import *
from runSLAM import run_slam_simulated
from optimization import cost_function
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
# M = len(landmarks)

# %% Initilize
Q = np.diag([0.2, 0.2, 0.001])  # TODO Best
Q = np.diag([0.001, 0.001, 0.0001])  # TODO Best
R = np.diag([0.0025, 0.003])  # TODO Best
R = np.diag([0.0015, 0.0006])  # TODO Best
# assert 0
do_asso = True

JCBBalphas = np.array(
    [10e-4, 10e-6]  # TODO,
)   # first is for joint compatibility, second is individual
# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.

alpha = 0.95
N = 200

eta_pred_init = poseGT[0]  # we start at the correct position for reference
P_pred_init = np.zeros((3, 3))
# allocate

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = True

args = (Q, R, JCBBalphas, eta_pred_init, P_pred_init,
        odometry, z, poseGT, N, alpha, do_asso, doAssoPlot)

retval = run_slam_simulated(*args)

(eta_pred, P_pred, eta_hat, P_hat, a, NIS, NISnorm, CI, CInorm, NEES) = retval

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

# %% Plotting of results
# movie = play_movie(pose_est, poseGT, lmk_est, landmarks, P_hat, N)
# plot_trajectory(pose_est, poseGT, P_hat, N)
plot_error(pose_est, poseGT, N)
plot_RMSE(pose_est, poseGT, N)
# plot_path(pose_est, poseGT, lmk_est_final, landmarks, P_hat, N)
# plot_NIS(NISnorm, CInorm, N)
# plot_NEES(NEES, alpha, N)
plt.show()
