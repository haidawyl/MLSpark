#!/usr/bin/python
#  -*- coding:utf-8 -*-

import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np


def plot_gallery(images, h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[:, i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title("Eigenface %d" % (i + 1), size=12)
        plt.xticks(())
        plt.yticks(())


if __name__ == "__main__":
    path = "D:/work/git/github/MLSpark/src/main/resources/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    ae = im.imread(path)
    plt.imshow(ae)
    plt.show()

    pcs = np.loadtxt("D:/work/git/github/MLSpark/src/main/resources/data/lfw/pc.csv", delimiter=",")
    print(pcs.shape)
