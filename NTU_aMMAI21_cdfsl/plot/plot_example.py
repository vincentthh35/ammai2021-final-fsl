import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np

COLORS = {
    'subspace': ['steelblue', 'cadetblue'],
}
N_POINT = 100

if __name__ == '__main__':
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(0,0)
    ax.axis["x"].set_axisline_style("->", size = 1.0)

    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("->", size = 1.0)

    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")

    xx = np.random.normal(0, 100, N_POINT)
    xy = np.random.normal(0, 100, N_POINT)

    yx = np.random.normal(0, 100, N_POINT)
    yy = np.random.normal(0, 100, N_POINT)

    plt.xlim(-250, 250)
    plt.ylim(-250, 250)

    plt.scatter(xx, xy, c='limegreen')
    plt.scatter(xy, yy, c='cornflowerblue')
    # plt.show()
    # plt.show()
    plt.savefig('before.png', dpi=300)
