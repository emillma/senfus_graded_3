import numpy as np
from typing import List, Optional
from matplotlib import pyplot as plt
from EKFSLAM import EKFSLAM
from tqdm import tqdm
from scipy.stats import chi2
from vp_utils import detectTrees,  odometry as get_odometry


def run_slam_simulated(Q, R, JCBBalphas, eta_pred_init, P_pred_init,
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
            eta_pred[k + 1], P_pred[k + 1] = slam.predict(
                eta_hat[k], P_hat[k], odometry[k])

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


def run_slam_real(Q, R, JCBBalphas, eta_pred_init, P_pred_init, car,
                  realSLAM_ws, N, alpha, do_asso=True, doAssoPlot=False):

    timeOdo = (realSLAM_ws["time"] / 1000).ravel()
    timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
    timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

    steering = realSLAM_ws["steering"].ravel()
    speed = realSLAM_ws["speed"].ravel()
    LASER = (
        realSLAM_ws["LASER"] / 100
    )  # Divide by 100 to be compatible with Python implementation of detectTrees
    La_m = realSLAM_ws["La_m"].ravel()
    Lo_m = realSLAM_ws["Lo_m"].ravel()

    K = timeOdo.size
    mK = timeLsr.size
    Kgps = timeGps.size
    sensorOffset = np.array([car.a + car.L, car.b])

    slam = EKFSLAM(Q, R, do_asso=do_asso, alphas=JCBBalphas,
                   sensor_offset=sensorOffset)

    # For consistency testing
    alpha = 0.05
    confidence_prob = 1 - alpha

    xupd = np.zeros((mK, 3))
    a = [None] * mK
    NIS = np.zeros(mK)
    NISnorm = np.zeros(mK)
    CI = np.zeros((mK, 2))
    CInorm = np.zeros((mK, 2))

    # Initialize state
    # you might want to tweak these for a good reference
    eta = np.array([Lo_m[0], La_m[1], 36 * np.pi / 180])
    P = np.zeros((3, 3))

    mk_first = 1  # first seems to be a bit off in timing
    mk = mk_first
    t = timeOdo[0]

    # %%  run
    N = 10000  # K = 61945 is max?

    doPlot = True

    lh_pose = None

    if doPlot:
        fig, ax = plt.subplots(num=1, clear=True)

        lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
        sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
        sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

    do_raw_prediction = True
    if do_raw_prediction:  # TODO: further processing such as plotting
        odos = np.zeros((K, 3))
        odox = np.zeros((K, 3))
        odox[0] = eta

        for k in range(min(N, K - 1)):
            odos[k + 1] = get_odometry(speed[k + 1],
                                       steering[k + 1], 0.025, car)
            odox[k + 1], _ = slam.predict(odox[k], P, odos[k + 1])

    for k in tqdm(range(N)):
        if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
            # Force P to symmetric: there are issues with long runs (>10000 steps)
            # seem like the prediction might be introducing some minor asymetries,
            # so best to force P symetric before update (where chol etc. is used).
            # TODO: remove this for short debug runs in order to see if there are small errors
            P = (P + P.T) / 2
            dt = timeLsr[mk] - t
            if dt < 0:  # avoid assertions as they can be optimized avay?
                raise ValueError("negative time increment")

            # ? reset time to this laser time for next post predict
            t = timeLsr[mk]
            odo = get_odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)  # TODO predict

            z = detectTrees(LASER[mk])
            eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)  # TODO update

            num_asso = np.count_nonzero(a[mk] > -1)

            if num_asso > 0:
                NISnorm[mk] = NIS[mk] / (2 * num_asso)
                CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                    2 * num_asso
                )
            else:
                NISnorm[mk] = 1
                CInorm[mk].fill(1)

            xupd[mk] = eta[:3]

            if doPlot:
                sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
                if len(z) > 0:
                    zinmap = (
                        rotmat2d(eta[2])
                        @ (
                            z[:, 0] *
                            np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                            + slam.sensor_offset[:, None]
                        )
                        + eta[0:2, None]
                    )
                    sh_Z.set_offsets(zinmap.T)
                lh_pose.set_data(*xupd[mk_first:mk, :2].T)

                ax.set(
                    xlim=[-200, 200],
                    ylim=[-200, 200],
                    title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
                )
                plt.draw()
                plt.pause(0.00001)

            mk += 1

        if k < K - 1:
            dt = timeOdo[k + 1] - t
            t = timeOdo[k + 1]
            odo = get_odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)
