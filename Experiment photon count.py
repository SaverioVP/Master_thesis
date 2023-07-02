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


Source_sphere = SourceCalculator("sphere3D", 256, 57.75, 57.75,
                                  constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                  sphere3D_radius_mm=6, sphere3D_center=np.asarray([127, 127, 127])
                                  )

# Source_ps = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57.75, size_z_mm=57.75,
#                                           constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
#                                           point_source_multiple_num_sources=1, 
#                                           point_source_multiple_coords=np.asarray([[127,127,127]])
#                                           )

# ==================================== Section: Simulation =========================================

simulation_obj_pc10e3= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**3,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

simulation_obj_pc10e4= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**4,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

simulation_obj_pc10e5= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**5,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

simulation_obj_pc10e6= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**6,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

simulation_obj_pc10e7= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**7,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

simulation_obj_pc10e8= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**8,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )
# ========================== Section: Reconstruction ===============================

test_recon_dist = np.linspace(166,178,7)

mlem_iters = 25

recon_10e3 = Reconstruction(
    sim_object=simulation_obj_pc10e3,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_10e4 = Reconstruction(
    sim_object=simulation_obj_pc10e4,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_10e5 = Reconstruction(
    sim_object=simulation_obj_pc10e5,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_10e6 = Reconstruction(
    sim_object=simulation_obj_pc10e6,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_10e7 = Reconstruction(
    sim_object=simulation_obj_pc10e7,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

recon_10e8 = Reconstruction(
    sim_object=simulation_obj_pc10e8,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=mlem_iters,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)



#recon.plot_reconstructed_slices()
recon_10e8.plot_3d_pyvista_single(recon_10e8.reconstructed_source_3D, 255)