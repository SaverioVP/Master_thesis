# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 13:13:01 2022

@author: saver
"""

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


coords_1 = [127,127,127]


source_coords_close = np.asarray([coords_1])


# Source_for_visualization = SourceCalculator("multisphere3D", 256, 57.57, 57.57,
#             constrain_to_bounds = None, z_scaling = False, allow_overlap = None,
#             multisphere3D_num_spheres = 1, 
#             multisphere3D_radii = np.asarray([2]), 
#             multisphere3D_sphere_coords = source_coords_close)


Source_sim_PS = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57, size_z_mm=150,
                                         constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                         point_source_multiple_num_sources=1, 
                                         point_source_multiple_coords=source_coords_close
                                         )

simulation_close = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**9,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sim_PS, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_3_points_close",
                              conv_mode="quick"
                              )


# recon_dist = np.linspace(170,173,4)

# recon_sim = Reconstruction(
#     sim_object=simulation_close,
#     distance_array=recon_dist,
#     MLEM_iters=2,
#     compute_3D=True,
#     post_processing=False,
#     conv_mode="quick"
# )

# recon_TOPAS = Reconstruction(
#     sim_object=TOPAS_sim_close,
#     distance_array=recon_dist,
#     MLEM_iters=2,
#     compute_3D=True,
#     post_processing=False,
#     conv_mode="quick"
# )

# recon_TOPAS.plot_3d_pyvista_multi(Source_for_visualization.source_array,
#                                   recon_sim.reconstructed_source_3D, 
#                                   255,255)


simulation_close.plot_detector_image()


