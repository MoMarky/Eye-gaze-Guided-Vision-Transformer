import os
import os.path as osp
import argparse
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """
    screen = cv2.imread(imagefile)
    # construct screen (black background)
    # screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    # if imagefile != None:
    #     # check if the path to the image exists
    #     if not os.path.isfile(imagefile):
    #         raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
    #     # load image
    #     img = cv2.imread(imagefile)
    #
    #     # width and height of the image
    #     w, h = len(img[0]), len(img)
    #     # x and y position of the image on the display
    #     x = int(dispsize[0] / 2 - w / 2)
    #     y = int(dispsize[1] / 2 - h / 2)
    #     # draw the image on the screen
    #     screen[y:y + h, x:x + w, :] += img

    # cv2.imshow('screen', cv2.resize(screen, None, fx=0.3, fy=0.3))
    # cv2.waitKey(0)
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax
def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M
def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5) 透明度
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]

    return_array = heatmap.copy()
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)
    # cv2.imshow('heatmap', cv2.resize(heatmap, None, fx=0.3, fy=0.3))

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    pyplot.clf()
    pyplot.close()

    return return_array


def run_draw_heatmap():
	img_root_path = "root path for your img"
    gaze_root_path = "root path of gaze.csv files"

    for gaze_csv_name in tqdm(os.listdir(gaze_root_path)):
    print(gaze_csv_name)
    gaze_lines = open(osp.join(gaze_root_path, gaze_csv_name), 'r').readlines()
    gaze_data = []
    for init_line in gaze_lines:
        tmp = init_line.split(',')
        gaze_data.append([int(tmp[0]), int(tmp[1]), 1])

    gray_img_name = '{}.jpg'.format(gaze_csv_name.split('.')[0])
    background_image_path = osp.join(img_root_path, gray_img_name)
    bk_img = cv2.imread(background_image_path)

    alpha = 0.5
    display_height, display_width, c = bk_img.shape
    ngaussian = 500
    sd = None

    array_save_root = r'Save root\{}_{}_{}_heatmap\hm_array'.format(alpha, ngaussian, 'None' if sd is None else sd)
    os.makedirs(array_save_root, exist_ok=True)

    heat_map_save_root = r'Save root\{}_{}_{}_heatmap\hm_img'.format(alpha, ngaussian, 'None' if sd is None else sd)
    os.makedirs(heat_map_save_root, exist_ok=True)
    save_heatmap_path = osp.join(heat_map_save_root, gray_img_name)

    heatmap_array = draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=save_heatmap_path,
                 imagefile=background_image_path, gaussianwh=ngaussian, gaussiansd=sd)

    np.save(osp.join(array_save_root, '{}.npy'.format(gaze_csv_name.split('.')[0])), heatmap_array)



