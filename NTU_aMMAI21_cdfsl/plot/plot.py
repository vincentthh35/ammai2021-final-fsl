import matplotlib.pyplot as plt
import os
import numpy as np

METHODS = ['subspace', 'subspace_plus']
TASKS = ['fsl', 'cdfsl-multi']
COLORS = {
    'subspace': ['steelblue', 'cadetblue'],
    'subspace_plus': ['sandybrown', 'coral'],
}

if __name__ == '__main__':
    fig, ax = plt.subplots(len(TASKS))
    for i, task in enumerate(TASKS):
        for method in METHODS:
            try:
                with open(os.path.join('..', 'logs', f'{method}_{task}', 'train_acc.txt'), 'r') as f:
                    lines = f.readlines()
                    lines = list(map(lambda s: s.rstrip().split(': '), lines))
                    data = np.array(list(map(lambda xy: [int(xy[0]), float(xy[1])], lines)))
                    if task == 'fsl':
                        ax[i].plot(data[:, 0], data[:, 1], COLORS[method][0], label=f'{method}')
                        ax[i].legend()
                    elif task == 'cdfsl-multi':
                        imgnet = np.arange(0, 300, 2)
                        crop = np.arange(1, 300, 2)
                        ax[i].plot(imgnet, data[imgnet, 1], COLORS[method][0], label=f'{method}: ImageNet')
                        ax[i].plot(crop, data[crop, 1], COLORS[method][1], label=f'{method}: CropDisease')
                        ax[i].legend()
            except Exception as e:
                print(e)
    ax[0].set_title('Training Accuracy of Different Methods')
    # plt.show()
    plt.savefig('training_curve.png', dpi=300)
