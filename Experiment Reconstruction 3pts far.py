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

file_str_far = "npy_files/Saverio_3p_far.npy"
photon_count_far = np.sum(np.load(file_str_far))

R_far_z = 150 / 256
R_far_xy = 57.75 / 256
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
print(source_coords_far)


Source_for_visualization = SourceCalculator("multisphere3D", 256, 57.75, 150,
            constrain_to_bounds = None, z_scaling = False, allow_overlap = None,
            multisphere3D_num_spheres = 3, 
            multisphere3D_radii = np.asarray([3, 3, 3]), 
            multisphere3D_sphere_coords = source_coords_far)


Source_sim_far = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57.75, size_z_mm=150,
                                         constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                         point_source_multiple_num_sources=3, 
                                         point_source_multiple_coords=source_coords_far
                                         )

simulation_far = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=photon_count_far,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sim_far, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_3_points_far",
                              conv_mode="quick"
                              )

TOPAS_sim_far = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                                  a=172, b=42,  # Setup geometry
                                  PSF_array_shape=512, Photon_count_total=photon_count_far,  # Simulation params
                                  Source=None, source_filepath="",  # if given, will load a source object
                                  # if given, will load a detector image instead of running the Sim
                                  detector_image_filepath=file_str_far,
                                  simulation_name=file_str_far,
                                  conv_mode="quick"
                                  )


# test_recon_dist = np.asarray([])
recon_dist = np.linspace(100,244,13)

recon_sim = Reconstruction(
    sim_object=simulation_far,
    distance_array=recon_dist,
    MLEM_iters=5,
    size_z_mm = 150,
    source_size_px = 256,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_TOPAS = Reconstruction(
    sim_object=TOPAS_sim_far,
    distance_array=recon_dist,
    MLEM_iters=5,
    size_z_mm = 150,
    source_size_px = 256,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)
#TOPAS_sim_far.plot_compare_detector_images(simulation_far.Detector_image_noisey, TOPAS_sim_far.Detector_image_noisey)
#recon_TOPAS.plot_3d_pyvista_single(recon_sim.reconstructed_source_3D, 400)
recon_TOPAS.plot_3d_pyvista_single(Source_for_visualization.source_array, 255)