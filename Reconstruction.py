# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:15:46 2022

@author: saver

This class handles the reconstruction part of the simulation. 
It is completely decoupled from the "Simulation" class, which handles the forward projection.
It takes a Simulation class instance as sim_object and will use that to reconstruct. 
The only information it pulls from the sim_object is the forward projection, so that recon remains decoupled.

"""
import numpy as np
import matplotlib.pyplot as plt
import time
from HelperFuncs import HelperFuncs
import plotly.io as pio
pio.renderers.default='browser'

class Reconstruction:
    def __init__(self, 
                 sim_object,
                 distance_array,
                 MLEM_iters,
                 size_z_mm,
                 source_size_px,
                 img_size_px = 256,
                 epsilon = 10**-8,
                 compute_3D = False,
                 conv_mode = "quick"
                 
                 ): 
        """
        Uses the 3D MLEM image reconstruction algorithm given in Mu 2006
        to reconstruct slices of a 3D object from a single detector image.
        ----------------------------------------------------------------
        
        Parameters
        ----------
        detector_image : 2d numpy array
            the detector image from which to reconstruct slices
        distance_array : 1d numpy array (distances in mm)
            array of Obj to Mask distances for each slice. eg 180 will reconstruct a slice at 180 mm
        num_iters: int
            how many MLEM iterations to perform for each slice. Takes about 5 s per iter per slice. 
            To do it in less than 5 minutes, use 10 to 20 slices, up to 35 iters
        (optional) img_size_px: int
            how large each slice array will be, eg 256 -> 256x256 image
            NOT TESTED FOR ANY OTHER SIZES
        (optional) epsilon: float
            small value for replacing negative or 0 values. default = 10e-8
        (optional) compute_3D: bool
            whether or not to compute the 3D reconstruction. True by default     
        """
       
        # Simulation class holds all the data for the forward projection, ie 3D detector image, and some methods to use
        self.sim_object = sim_object
        
        # Reconstruction params
        self.distance_array = distance_array  # At which distance do you want to reconstruct each slice?
        self.MLEM_iters = MLEM_iters  # How many iterations to use
        self.size_z_mm = size_z_mm  # This is used when recreating the 3D reconstruction in "recon_3D()"
        self.source_size_px = source_size_px #This is used when recreating the 3D reconstruction in "recon_3D()"
        self.img_size_px = img_size_px  # currently only 256x256
        self.epsilon = epsilon
        self.conv_mode = conv_mode # leave this at "quick"
        
        # Arrays for storing reconstruction        
        self.reconstructed_slices = self.reconstruct_new()
        
        
        if compute_3D == True:
            self.reconstructed_source_3D = self.recon_3D()
        else:
            self.reconstructed_source_3D = None    
        
    def reconstruct_new(self):
        """
        This is the main reconstruction function. 
        Reference Mu and Liu 2006 for the 3D MLEM algorithm
        Returns 3D array of reconstructed slices, eg 256x256xN
        """
        
        # Some assignments to make coding easier
        distance_array = self.distance_array
        num_iters = self.MLEM_iters
        img_size_px = self.img_size_px
        epsilon = self.epsilon
        detector_image = self.sim_object.Detector_image_noisey
        
        # Initialize some values
        N = distance_array.shape[0]
        f_array = np.ones((img_size_px,img_size_px, N))  # This is the initial guess, currently only using all ones
        PSF_array = []
        
        # Precalculate array of PSFs for each desired reconstruction slice.
        z_sum = 0
        for i in range(N):
            if self.conv_mode == "default":
                PSF_array.append(self.sim_object.PSF_raycast_from_mask_keep_resolution(distance_array[i]))
            elif self.conv_mode == "quick":
                PSF_array.append(HelperFuncs.pad_or_crop(self.sim_object.PSF_raycast_from_mask_keep_resolution(distance_array[i]),512))
            # Calculate the z_sum term in denominator outside of brackets
            
            # Find scalar sum of all PSFs for the end
            z_sum += np.sum(PSF_array[i])
            
        # Main loop
        start_time = time.process_time()
        
        for k in range(num_iters):  
            for i in range(N):  # for each iteration, need to go through each slice and subtract each "out of focus" slice

                # Reset the sum in numerator
                bracket_sum = np.zeros((img_size_px,img_size_px))
                
                # Perform the sum in numerator
                for j in range(N): 
                    if i != j:
                        # Find estimated forward projections for each out of focus slice and sum together
                        bracket_sum += HelperFuncs.custom_tf_conv2d(f_array[:,:,j], PSF_array[j])

                # Reset the numerator for this slice
                numerator = np.zeros((img_size_px,img_size_px))
                
                # Calculate the numerator
                numerator = detector_image - bracket_sum  # subtract out of focus contribution to p
                
                # Replace all negative or 0 values with epsilon to avoid negative pixels 
                numerator[numerator < epsilon] = epsilon
                
                # This is the Forward Projection
                denominator = HelperFuncs.custom_tf_conv2d(f_array[:,:,i], PSF_array[i]) + epsilon 
                
                # Calculate the right hand side of the value in brackets
                bracket_val_right= numerator/denominator  

                # Calculate the entire value in the brackets
                bracketval_total = HelperFuncs.custom_tf_conv2d(np.rot90(bracket_val_right,2), PSF_array[i]) # rotating 180 degrees to represent the correlation

                # Perform element-wise division then element-wise multiplication
                f_array[:,:,i] = (f_array[:,:,i] / z_sum) * bracketval_total  # formula in paper is unclear if z_sum is scalar sum of PDFs, but it only seemed to work under that assumption.
                    
        
        final_time = -1*(start_time - time.process_time())
        
        print("Reconstruction of ", N, " slices with ", num_iters, " iterations completed in ", final_time, "seconds with conv mode = ", self.conv_mode)
        return f_array # Return the array of reconstructed slices
  
 
    def recon_3D(self):
        """
        Concatenates the stack of reconstructed slices into a 3D array, based on their distance from the mask.
        Will convert from mm to pixel index along the z direction. Uses given size_z_mm and source_size_px, which should match the source
        
        
        Only call this after you have done the reconstruction or else nothing will happen. 
        This is called automatically if compute_3D is set to True
        """

        resolution_z = self.size_z_mm / self.source_size_px
        
        # Find the extents of the source cube in mm
        source_span_start = self.sim_object.a - self.size_z_mm/2
        source_span_end = self.sim_object.a + self.size_z_mm/2
        
        
        # Make sure all the requested slices are in bounds
        if np.amin(self.distance_array) < source_span_start or np.amax(self.distance_array) > source_span_end:
            print("recon_3D FAILED: requested slices are out of bounds of the source cube in z direction")
        
        recon_arr = self.reconstructed_slices
        reconstructed_source_3D = np.zeros((self.source_size_px, self.source_size_px, self.source_size_px))
        
        # Now just have to stuff recon_arr into recon_source_3D
        for i in range(recon_arr.shape[2]):  # this loops over each slice
            z_val_px = np.round((self.distance_array[i] - source_span_start)/ resolution_z).astype(int)
            
            # Add the slice to the 3D source object
            reconstructed_source_3D[:,:,z_val_px] = np.rot90(recon_arr[:,:,i],2)  # No idea why rotation by 180 is required but it's backwards if you don't
            
        return reconstructed_source_3D

#==================================SECTION: Visualization Methods =======================================
        
    def plot_reconstructed_slices(self, slicenum):
        # Plots the stack of reconstructed slices
        
        fig, ax = plt.subplots(2, 2, figsize = (7.5,7.0))

        im = ax[0,0].imshow(self.reconstructed_slices[:,:,slicenum])

        fig.colorbar(im, ax=ax[0,0], fraction=0.046, pad=0.04)

    def plot_3d_pyvista_single(self, arr, multiplier):
        """
        This is the main way to do 3D visualization. array is 3D numpy binary array to be visualized
        multiplier is required to get the array values within 1 - 255 or else it just wont show up properly
        """
        import pyvista as pv
        
        # stuff to make it pretty
        sargs = dict(
            title_font_size=20,
            label_font_size=16,
            shadow=True,
            n_labels=3,
            italic=True,
            fmt="%.1f",
            font_family="arial",
            width = 0.3,
            position_x = 0.15,
            position_y= 0.01
        )
        
        # initialize pyvista plotter object
        p = pv.Plotter(shape=(1, 1)) 
        
        values = arr*multiplier/ np.amax( arr)  # multiply the array by multiplier
        
        #add the pyvista volume to axes
        p.add_volume(values, cmap = "bone", opacity = "sigmoid_6", scalar_bar_args=sargs)
        
        #show the axes
        p.show_axes()
        p.show_bounds(xlabel='X',ylabel='Y',zlabel='Z', grid='front', all_edges=True, location='outer')
        
        # Find optimal camera view and keep it same for each plot
        p.view_xy()
        p.camera.up = (0.0,-1.0,0.0)
        p.camera.azimuth += 220
        p.camera.elevation += 20

        p.add_text("3D array")
        
        p.show()
    