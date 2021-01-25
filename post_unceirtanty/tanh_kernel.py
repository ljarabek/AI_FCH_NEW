import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def tanh_filter(depth=32, k=32, dullness=2):
    blank = np.zeros(shape=(k, k))
    for i in range(k):
        blank[i, :] = np.arange(0, k)
    blank -= k / 2
    blank /= dullness
    blank = np.tanh(blank)
    # plt.imshow(blank)
    # plt.show()
    blank_t = np.transpose(blank)
    filter = np.array([blank, blank_t])
    filter = np.expand_dims(filter, -1)
    filter = np.tile(filter, (1, 1, 1, 32))  # TODO: kter axis je anteroposteriorni?
    filter = np.expand_dims(filter, 1)
    return filter  # output of shape (out_ch, in_ch, h,w,d)


if __name__ == "__main__":
    mpl.use("Qt5Agg")
    print(tanh_filter().shape)
