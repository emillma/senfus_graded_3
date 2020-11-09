from matplotlib import pyplot as plt


def mes_diff(z, zpred):
    zpred = zpred.reshape(-1, 2)
    z = z.reshape(-1, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(z[:, 1], z[:, 1], c='r')
    ax.scatter(zpred[:, 1], zpred[:, 1], c='g')
    plt.show()
    return fig
