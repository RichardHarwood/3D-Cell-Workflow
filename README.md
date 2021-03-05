
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

Next we want to set a working directory, an image output and where the images are stored and define what sample we are working on.

```python 
os.chdir('F:/Organelle Distances/Python Output/')
output ="F:/Organelle Distances/Python Output/"
image_dir = "z:/ALLTIFS3DMIT/"
sample="D1_C1_"
```
Next we bring in the images and reduce them - reducing them is optional and I dont for any quanitativite output. But when I am working on data vis pipelines I like working with small(ish) arrays

```python 
chl_name='D1C1CHL.tif';mit_name='D1C1MIT.tif';cell_w_org_name="D1C1VAC.tif"
cell_name="D1C1CELL.tif";air_name="D1C1AIR.tif";adj_name="D1C1ADJ.tif"

chl=io.imread(image_dir + chl_name)
chl=zoom(chl, (0.5, 0.5, 0.5)) #This is the same as Image > adjust >size in FIJI
mit = io.imread(image_dir + mit_name)
mit=zoom(mit, (0.5, 0.5, 0.5)) #This is the same as Image > adjust >size in FIJI
air = io.imread(image_dir + air_name)
air=zoom(air, (0.5, 0.5, 0.5)) #This is the same as Image > adjust >size in FIJI
adj = io.imread(image_dir + adj_name)
adj=zoom(adj, (0.5, 0.5, 0.5)) #This is the same as Image > adjust >size in FIJI
```
zoom is just the same as adjust size in ImageJ. Our initial voxel size for these e.g. images was 6nm, 6nm 50nm. But before segmenting I reduced them to 20nm,20nm 50nm. At this stage I was using a core facility computer. For this analysis I reduce them to 40nm, 40nm, 100nm.
Define voxel values in um

```python 
X=0.04
Y=0.04
Z=0.1
```

The above images are binary- a voxel is an organelle or tissue or it is not. The folowing code creates a 3D mesh. H This runs the "marching cubes" algorithim which "extracts a polygonal mesh of an isosurface from a three-dimensional discrete scalar field (sometimes called a voxel). 
Here are the resources I used https://en.wikipedia.org/wiki/Marching_cubes https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html

Create the mesh

```python 
vertices, faces, normals, values = marching_cubes_lewiner(chl, level=None,
                                                          spacing=(Z, X,Y), gradient_direction='descent', step_size=1, 
                                                          allow_degenerate=True, use_classic=False) 
chlmesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        chlmesh .vectors[i][j] = vertices[f[j],:]
```
save the mesh as an STL

```python 
chlmesh.save(sample +'chl3D.stl')
```
and then read it in again 
```python 
CHLstl = pv.read(output+sample +"chl3D.stl")
```

Now we can see the chloroplasts in 3D - Note that the a HQ image is saved but i am showing GIFs here cause they are nicer :) the code to make the gifs is at the end of the readme 
```python
pv.set_plot_theme("document")
p = pv.Plotter()
p.add_mesh(CHLstl, color="green", opacity=1)
p.window_size = 500, 500
p.show(screenshot=sample+"   Image1.tiff", window_size=[2400,2400])
```


Now we do the same for the mitochondria 
```python 
vertices, faces, normals, values = marching_cubes_lewiner(mit, level=None, spacing=(Z, X,Y),
                                                          gradient_direction='descent', step_size=1, allow_degenerate=True,
                                                          use_classic=False) 
mitmesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        mitmesh .vectors[i][j] = vertices[f[j],:]
```

```python 
mitmesh.save(sample +'mit3D.stl')
MITstl = pv.read(output+sample +"mit3D.stl")
```
```python 
pv.set_plot_theme("document")
p = pv.Plotter()
p.add_mesh(MITstl, color="red", opacity=1)
p.window_size = 400, 400
p.show(screenshot=sample+"   Image2.tiff", window_size=[2400,2400])
```
![](content/D1_C1_MIT.gif)
