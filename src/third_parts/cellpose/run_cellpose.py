import os
import numpy as np
import time
import sys
import random
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import shutil

from cellpose import models, core

# files = ['/home/zaiwang/workspace/cellpose/demos/data/Lung5_Rep1/Lung5_Rep1-RawMorphologyImages/'
         # '20210907_180607_S1_C902_P99_N99_F032_Z003.TIF']

# files = ['/home/zaiwang/workspace/cellpose/demos/data/CellComposite_F003.jpg']
files = ['/home/zaiwang/workspace/cellpose/demos/data/000_img.tif']

imgs = []

for f in files:
    im = skimage.io.imread(f)
    print(im.shape)
    n_dim = len(im.shape)
    dim = im.shape
    print(n_dim)
    channel = min(dim)  # channel will be dimension with min value usually
    channel_position = dim.index(channel)
    print(channel_position)
    if n_dim == 3 and channel_position == 0:
        # print(dim)
        im = im.transpose(1, 2, 0)
        dim = im.shape
        # print("Shape changed")
        # print(dim)
    imgs.append(im)

nimg = len(imgs)
print("No of images loaded are: ", nimg)

random_idx = random.choice(range(len(imgs)))
print(random_idx)
x = imgs[random_idx]
n_dim = len(x.shape)
if n_dim == 3:
    channel_image = x.shape[2]
    print(channel_image)
    fig, axs = plt.subplots(1, channel_image, figsize=(12, 5))
    for channel in range(channel_image):
        axs[channel].imshow(x[:, :, channel])
        axs[channel].set_title('Channel ' + str(channel + 1), size=5)
        axs[channel].axis('off')
    fig.tight_layout()
    plt.show()

Model_Choice = "Cytoplasm"
model_choice = Model_Choice

print("Using model ", model_choice)

Channel_for_segmentation = "2"
segment_channel = int(Channel_for_segmentation)

Use_nuclear_channel = False
Nuclear_channel = "3"  # @param[1,2,3]
nuclear_channel = int(Nuclear_channel)

omni = False
Diameter = 0
diameter = Diameter

if model_choice == "Cytoplasm":
    model_type = "cyto"

elif model_choice == "Cytoplasm2":
    model_type = "cyto2"

elif model_choice == "Cytoplasm2_Omnipose":
    model_type = "cyto2_omni"

elif model_choice == "Bacteria_Omnipose":
    model_type = "bact_omni"
    diameter = 0

elif model_choice == "Nuclei":
    model_type = "nuclei"

if model_choice not in "Nucleus":
    if Use_nuclear_channel:
        channels = [segment_channel, nuclear_channel]
    else:
        channels = [segment_channel, 0]
else:  # nucleus
    channels = [segment_channel, 0]

channels = [1, 0]

use_GPU = True
model = models.Cellpose(gpu=use_GPU, model_type=model_type)

if diameter == 0:
    diameter = None

from skimage.util import img_as_ubyte

Image_Number = 1  # @param {type:"number"}
Image_Number -= 1  # indexing starts at zero
# print(Image_Number)
Diameter = 0  # @param {type:"number"}
# Flow_Threshold=0.4#@param {type:"number"}
Flow_Threshold = 0.3  # @param {type:"slider", min:0.1, max:1.1, step:0.1}
flow_threshold = Flow_Threshold

# Cell_Probability_Threshold=-5#@param {type:"number"}
# Using slider to restrict values
Cell_Probability_Threshold = -1  # @param {type:"slider", min:-6, max:6, step:1}
cellprob_threshold = Cell_Probability_Threshold

diameter = Diameter
if diameter == 0:
    diameter = None
if Image_Number == -1:
    Image_Number = 0
    # print("Image_Number is set to zero, opening first image.")
try:
    image = imgs[Image_Number]
except IndexError as i:
    print("Image number does not exist", i)
    print("Actual no of images in folder: ", len(imgs))
print("Image: %s" % (os.path.splitext(os.path.basename(files[Image_Number]))[0]))
img1 = imgs[Image_Number]



import cv2

print("begin to evaluation....")
print(np.shape(img1), 'img1 shape')

masks, flows, styles, diams = model.eval(img1, diameter=diameter, flow_threshold=flow_threshold,
                                         cellprob_threshold=cellprob_threshold, channels=channels)
# print(masks)
# print(flows)
# print(styles)
# print(diams)
print("evaluation done!")
# DISPLAY RESULTS
from cellpose import plot

maski = masks
flowi = flows[0]

# convert to 8-bit if not so it can display properly in the graph
if img1.dtype != 'uint8':
    img1 = img_as_ubyte(img1)

fig = plt.figure(figsize=(24, 8))
plot.show_segmentation(fig, img1, maski, flowi, channels=channels)
plt.tight_layout()
plt.show()
