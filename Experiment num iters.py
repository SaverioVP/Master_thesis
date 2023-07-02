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

# ==================================== Section: Simulation =========================================

simulation_obj= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**9,  # Simulation params
                              # if given, will load a source object
                              Source=Source_sphere, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation_sphere",
                              conv_mode="quick"
                              )

#simulation_obj.plot_source3D(Source_sphere, vis_mode="voxels", swap_points=True, elevation=20, azimuth=120)
#simulation_obj.plot_detector_image()
#simulation_obj.plot_slices()
# ========================== Section: Reconstruction ===============================

# test_recon_dist = np.linspace(

test_recon_dist = np.asarray([166,172,178])

recon = Reconstruction(
    sim_object=simulation_obj,
    distance_array=test_recon_dist,
    size_z_mm = 57.75,
    source_size_px = 256,
    MLEM_iters=10,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

# recon.plot_reconstructed_slices()
#recon.plot_3d_pyvista_single(recon.reconstructed_source_3D, 255)

# ================ Stuff for Discussion =================
recon.plot_central_recon_slice()