from Reconstruction import Reconstruction
from SimulationManager import Simulation_Manager
from Simulation import Simulation
from DataCreator import Data_Creator
from Detector import DetectorCalculator
from Source import SourceCalculator
from Mask import MuraMaskCalculator
from HelperFuncs import HelperFuncs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import random
import kaleido
from stl import mesh
import trimesh
from PIL import Image, ImageFont, ImageDraw
import cv2
import time
import plotly.graph_objects as go
import plotly.express as px
import io
import matplotlib
from typing import Type
import scipy.signal as signal
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


Mask = MuraMaskCalculator(31)
Detector = DetectorCalculator()



R_far_z = 150 / 256
R_far_xy = 57 / 256
coords_far1 = [np.round(127 - 14.25/R_far_xy).astype(int)-15,
           np.round(127 - 14.25/R_far_xy).astype(int)-15,
           np.round(127 - 72/R_far_z).astype(int)]

coords_far2 = [127 + 20, 
               127 + 20, 
               127]

coords_far3 = [np.round(127 - 14.25/R_far_xy).astype(int)+0, 
           np.round(127 + 14.25/R_far_xy).astype(int)-0, 
           np.round(127 + 72 /R_far_z).astype(int)]

source_coords_far = np.asarray([coords_far1, 
                            coords_far2, 
                            coords_far3])

Source_ps = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57.75, size_z_mm=150,
                                          constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                          point_source_multiple_num_sources=3, 
                                          point_source_multiple_coords=source_coords_far
                                          )


# ==================================== Section: Simulation =========================================
simulation_obj= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**8,  # Simulation params
                              # if given, will load a source object
                              Source=Source_ps, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_mu",
                              conv_mode="quick"
                              )
#plt.imshow(Mask.mask_array[50:70,50:70], cmap = "gray")
center = int(499/2)
plt.imshow(simulation_obj.PSFs[1][0:160,0:160])
