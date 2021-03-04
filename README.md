
# This notebook will outline a few really cool image analysis and image visualiation techniques in Python

Firstly I will show how to automate visuliations
I think this apporach is really cool because it creates STLS from the image stacks then renders these - for me this means endless possiblities (e.g. VR - 3D printing)
Here is how you make a STL from an image stack
Its likely that not all the code is useful so to save this been a huge document chunks can be displayed by hitting toggle code
Load Packages

```python 
In [1]:
from skimage import img_as_ubyte, img_as_bool, img_as_float32
import sys
import os
import numpy as np
from pandas import DataFrame
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
import skfmm
import skimage.io as io
from skimage import img_as_ubyte
from skimage.util import invert
from skimage.measure import label, regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
import time
from tqdm import tqdm
import joblib
import multiprocessing
import cv2
from os import listdir
import matplotlib.pyplot as plt 
#import openpnm as op
#import porespy
#import pytrax as pt
from scipy import ndimage
from stl import mesh
#import visvis as vv 
from scipy.ndimage import zoom
import pyvista as pv
from scipy.spatial import KDTree
from skimage.external import tifffile as tif
from pyvista import examples
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import *
import vedit
import logging
import future
import ffmpeg
from skimage.morphology import binary_dilation 
from skimage.morphology import square
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
```
