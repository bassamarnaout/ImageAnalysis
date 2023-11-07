import math
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib import colors

from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import hsv_to_rgb


####################################################################################################
# Function: plotImageIntoRGBSpace
# It plots the image into RGB space
####################################################################################################
def plotImageIntoRGBSpace(img):
    # Plotting the image on 3D plot
    r, g, b = cv2.split(img)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(
        r.flatten(),g.flatten(),b.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


####################################################################################################
# Function: euclid_distance
# It Computes the euclidean distance between two points

# Input Parameter:
# param-x: first point
# param-xi : second point
#
# Returns : calculated distance value
####################################################################################################
def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))



####################################################################################################
# Function: gaussian_kernel
# It Computes the gussian value between two points

# Input Parameter:
# param-distance: distance between tow points
# param-bandwidth : bandwidth of the kernel
#
# Returns--val : calculated gussian value
####################################################################################################
def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val


####################################################################################################
# Function to convert number into string
# Switcher is dictionary data type here
####################################################################################################
def nameOfImageFile(imageNumber):
    switcher = {

        1: 'image1.ppm',
        2: 'image2.ppm',
        3: 'image3.ppm',
        4: 'image4.ppm',
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(imageNumber, "nothing")




def show_images(images, rows=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    rows (Default = 1): Number of rows in figure (number of columns is
                        set to np.ceil(n_images/float(rows))).

    titles: List of titles corresponding to each image. Must have
            the same length as images.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images / float(rows)), n + 1)
        height, width = image.shape[:2]
        if image.ndim == 2 and width >1: # plotting if image
            plt.gray()
            plt.imshow(image,interpolation = 'nearest')
        elif image.ndim == 2 and width ==1: # plotting of Histogram
            plt.plot(image)

        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()