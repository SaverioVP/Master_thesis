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

R_close = 57.75 / 256

Mask = MuraMaskCalculator(31)
Detector = DetectorCalculator()

npy_file_7 = "Saverio_3p_close_2.npy"
file_str_close = "npy_files/" + npy_file_7

photon_count_close = np.sum(np.load(file_str_close))


coords_close1 = [np.round(127 - 14.25/R_close).astype(int),# -2
           np.round(127 - 14.25/R_close).astype(int), #-2
           np.round(127 - 10.02/R_close).astype(int)]

coords_close2 = [127, 127, 127]

coords_close3 = [np.round(127 - 14.25/R_close).astype(int), #+5 
           np.round(127 + 14.25/R_close).astype(int), #-3 
           np.round(127 + 10.02/R_close).astype(int)]

source_coords_close = np.asarray([coords_close1, 
                            coords_close2, 
                            coords_close3])
print(source_coords_close)


Source_for_visualization = SourceCalculator("multisphere3D", 256, 57.75, 57.75,
            constrain_to_bounds = None, z_scaling = False, allow_overlap = None,
            multisphere3D_num_spheres = 3, 
            multisphere3D_radii = np.asarray([1.5, 1.5, 1.5]), 
            multisphere3D_sphere_coords = source_coords_close)


Source_sim_close = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57.75, size_z_mm=57.75,
                                         constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                         point_source_multiple_num_sources=3, 
                                         point_source_multiple_coords=source_coords_close
                                         )

simulation_close = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=photon_count_close,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sim_close, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_3_points_close",
                              conv_mode="quick"
                              )

TOPAS_sim_close = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                                  a=172, b=42,  # Setup geometry
                                  PSF_array_shape=512, Photon_count_total=photon_count_close,  # Simulation params
                                  Source=None, source_filepath="",  # if given, will load a source object
                                  # if given, will load a detector image instead of running the Sim
                                  detector_image_filepath=file_str_close,
                                  simulation_name=file_str_close,
                                  conv_mode="quick"
                                  )



recon_dist = np.linspace(162,182,11)

recon_sim = Reconstruction(
    sim_object=simulation_close,
    distance_array=recon_dist,
    MLEM_iters=5,
    size_z_mm = 57.75,
    source_size_px = 256,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

# recon_TOPAS = Reconstruction(
#     sim_object=TOPAS_sim_close,
#     distance_array=recon_dist,
#     MLEM_iters=5,
#     size_z_mm = 57.75,
#     source_size_px = 256,
#     compute_3D=True,
#     post_processing=False,
#     conv_mode="quick"
# )
#TOPAS_sim_close.plot_compare_detector_images(simulation_close.Detector_image_noisey, TOPAS_sim_close.Detector_image_noisey)
#recon_sim.plot_3d_pyvista_single(recon_sim.reconstructed_source_3D, 400)
recon_sim.plot_3d_pyvista_single(Source_for_visualization.source_array, 255)