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
#Sources: 3d Point source in center, Sphere, Multisphere, Letters, all with 10**8 photon count


Mask = MuraMaskCalculator(31)
Detector = DetectorCalculator()

# Source_ps = SourceCalculator(source_type="point_source_multiple", size_px=256, size_xy_mm=57.75, size_z_mm=57.75,
#                                           constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
#                                           point_source_multiple_num_sources=1, 
#                                           point_source_multiple_coords=np.asarray([[127,127,127]])
#                                           )

# Source_sphere = SourceCalculator("sphere3D", 256, 57.75, 57.75,
#                                   constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
#                                   sphere3D_radius_mm=6, sphere3D_center=np.asarray([127, 127, 127])
#                                   )


# multisphere_coords = np.asarray([[ 62 , 62 , 83], [127, 127, 127], [ 69, 187, 171]])
# Source_multisphere = SourceCalculator("multisphere3D", 256, 57.75, 57.75,
#                                   constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
#                                   multisphere3D_num_spheres = 3,
#                                   multisphere3D_radii = np.asarray([3,3,3]),
#                                   multisphere3D_sphere_coords = multisphere_coords
#                                   )    

# Source_mu = SourceCalculator(source_type="mu_tubes", size_px=256, size_xy_mm=57.75, size_z_mm=57.75,
#                                           constrain_to_bounds=None, z_scaling=False, allow_overlap=None
#                                           )

Source_letters = SourceCalculator(source_type="letters3D", size_px=256, size_xy_mm=57.75, size_z_mm=57.75,
                                          constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                          letters3D_string = '>H',
                                          letters3D_z_depths_mm = np.asarray([10, 50]),
                                          letters3D_z_thicknesses_mm = np.asarray([4, 4])
                                          )



# ==================================== Section: Simulation =========================================
simulation_obj= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**10,  # Simulation params
                              # if given, will load a source object
                              Source=Source_letters, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              detector_image_filepath="",
                              simulation_name="Simulation",
                              conv_mode="quick"
                              )


#simulation_obj.plot_detector_image_noisefree()
#simulation_obj.plot_detector_image_slices()
#simulation_obj.plot_slices()
# ========================== Section: Reconstruction ===============================


test_recon_dist = np.linspace(154,194,9)
#recon_dist_lazy = np.linspace(142,200,2)

recon = Reconstruction(
    sim_object=simulation_obj,
    distance_array=test_recon_dist,
    MLEM_iters=35,
    size_z_mm = 57.75,
    source_size_px = 256,
    compute_3D=True,
    post_processing=False,
    conv_mode="quick"
)

simulation_obj.plot_detector_image()
#Source_letters.plot_source_slices_2D(test_recon_dist , 172)
#recon.plot_3d_pyvista_single(Source_letters.source_array, 255)
recon.plot_3d_pyvista_single(recon.reconstructed_source_3D, 165)
#recon.plot_reconstructed_slices()