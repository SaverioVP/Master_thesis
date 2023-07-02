# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:54:48 2022

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


Source_sphere = SourceCalculator("sphere3D", 50, 20, 20,
                                 constrain_to_bounds=None, z_scaling=False, allow_overlap=None,
                                 sphere3D_radius_mm=7, sphere3D_center=np.asarray([25, 25, 25])
                                 )


# simulation_obj = Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
#                             a=172, b=42,  # Setup geometry
#                             PSF_array_shape=512, Photon_count_total=10**5,  # Simulation params
#                             # if given, will load a source object
#                             Source=Source_sphere,
#                             # if given, will load source from a filepath
#                             source_filepath="",
#                             # if given, will load a detector image instead of running the Sim
#                             detector_image_filepath="",
#                             simulation_name="Simulation_sphere",
#                             conv_mode="quick"
#                             )


# simulation_obj.plot_detector_image_slices()
# simulation_obj.plot_detector_image()
# print("plotting 3D...")
Source_sphere.plot_source_slices(42)
