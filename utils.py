from math import ceil

import matplotlib.image as mpli
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_images(images, titles=None, cols=2, figsize=(10, 10), color='gray', fontsize=10):
    if titles is None:
        titles = [None for i in range(len(images))]
    rows = ceil(len(images) / cols)
    plt.figure(figsize=figsize)
    #plt.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.imshow(img, color)
        plt.xticks([])
        plt.yticks([])
    plt.show()