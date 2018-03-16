import math
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_images(num, x, y, path):  
    num_images = num
    indices = np.random.choice(list(range(len(x))), size=num_images, replace=False)

    # Obtain the images and labels
    images = x[indices]
    labels = y[indices]

    for i, image in enumerate(images):
        plt.rcParams["figure.figsize"] = [15, 5]
        plt.subplot(2, math.ceil(num_images/2.), i+1)
        plt.imshow(image)
        plt.title('%s' % class_names[labels[i]])

    plt.tight_layout()
    plt.savefig(path)
    plt.show()