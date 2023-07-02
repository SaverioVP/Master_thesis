# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:21:07 2022

@author: saver

Reconstruction of Tobi's last email to me. 
This file reconstructs the NPY file and also acts as a tutorial for loading a pre-generated forward projection
"""

from Reconstruction import Reconstruction
from Simulation import Simulation
from Detector import DetectorCalculator
from Source import SourceCalculator
from Mask import MuraMaskCalculator
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_input_file = np.load("SRP_TestInput_ORp2p98median.npy")  # Load the test forward projection
#test_input_file = np.load("Saverio_3p_close_2.npy")  # Load the test forward projection


# Load the 5 files to test

#test_input = np.squeeze(test_input_file)
test_input = np.squeeze(test_input_file[4,:,:,:])
# test_input_2 = np.squeeze(test_input_file[1])
# test_input_3 = np.squeeze(test_input_file[2])
# test_input_4 = np.squeeze(test_input_file[3])
# test_input_5 = np.squeeze(test_input_file[4])

# Task:
# The sources are centered at 172mm and have a z-span of roughly 22mm. 
# If you could reconstruct the first 5 images and resend me those as numpy files and also 
# screenshots from your nice pyvista visualization of them.


#First initialize the mask and detector objects
Mask = MuraMaskCalculator(31)
Detector = DetectorCalculator()

#==================================SECTION:  FORWARD PROJECTION======================================


#Create the simulation object using the loaded forward projection
simulation_obj= Simulation(Mask=Mask, Detector=Detector,  # Basic objects for CAI
                              a=172, b=42,  # Setup geometry
                              PSF_array_shape=512, Photon_count_total=10**9,  # Simulation params. photon count will be ignored if FP already given
                              # if given, will load a source object
                              Source=None, source_filepath="",
                              # if given, will load a detector image instead of running the Sim
                              given_detector_image=test_input,
                              simulation_name="Simulation",
                              conv_mode="quick"
                              )


#==================================SECTION: RECONSTRUCTION======================================
# initialize array of distances to reconstruct slices at.
test_recon_dist = np.linspace(172-10,172+10, 6)
test_recon_dist = np.asarray([172])

mlem_iters = 5  # how many MLEM iters to use. 10 is good for first guess

recon = Reconstruction(
    sim_object=simulation_obj,
    distance_array=test_recon_dist,
    size_z_mm = 24,  # should be slightly bigger than the span of distance array
    source_size_px = 256,  # always 256
    MLEM_iters=mlem_iters,
    compute_3D=True, # make 3D automatically
    conv_mode="quick"  # use quick conv method
)

#==================================SECTION: VISUALIZATION======================================
recon.plot_reconstructed_slices(0)
recon.plot_3d_pyvista_single(recon.reconstructed_source_3D, 100)
















