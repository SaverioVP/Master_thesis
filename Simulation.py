"""
Creates a simulation object and runs the CAI simulation. 
Will generate a forward projection of the given Source object using the given Mask and Detector
Forward projection stored in self.Detector_image_noisey 
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from HelperFuncs import HelperFuncs


class Simulation:
    def __init__(self, Mask, Detector,  # Basic objects for CAI
                 a, b,  # Setup geometry
                 PSF_array_shape, Photon_count_total,  # Simulation params
                 Source=None, source_filepath="",  # if given, will load a source object
                 given_detector_image = np.zeros((10,10)),  # if given, will load a detector image instead of running the Sim
                 simulation_name = "default simulation",
                 conv_mode = "quick"  
                 ):  # sad face
        

        # Organization
        self.simulation_name = simulation_name
        self.conv_mode = conv_mode # 'quick' or 'default'. see HelperFuncs.custom_tf_conv2d
        
        # CAI objects
        self.Mask = Mask
        self.Detector = Detector
        
        # Setup geometry
        self.a = a  # Object to Mask distance
        self.b = b  # Mask to detector distance
        self.z = a + b  # Object to detector distance
        self.m = (a + b) / a # Magnification Factor

        # Place to store slices of Source array
        self.num_slices = None  # how many slices in the source array?
        self.a_values = None  # distance of each slice from the mask
        self.slices = None  # the 2D array of the slices themselves

        # Place to store PSFs
        self.PSFs = None

        # Forward Projection params
        self.Photon_count_total = Photon_count_total  # The photon count for the ENTIRE 3D source image
        self.photon_alottment = None  # Array of photons alotted to each slice

        # Store the Forward Projection
        self.Detector_images_2D_noisefree = None  # the stack of forward projections for each slice
        self.Detector_image_noisefree = None  # the summed forward proj of entire 3D object, noisefree
        
        #This is the full detector image, which will be set below
        self.Detector_image_noisey = None
        
        #If detector image is given, use it and save it as detector image and thats it
        if np.sum(given_detector_image) != 0: 
            print("creating simulation with given detector image")
            self.Detector_image_noisey = given_detector_image
            
        else: #if not, then do all the rest below
            if Source != None:  #if a source object is given in code, use that
                print("creating simulation with given source object")    
                self.Source = Source
            else:  # not given, use the given filename to load a source object 
                print("creating simulation with given filepath")    
                with open(source_filepath, "rb") as fileObject:
                    self.Source = pickle.load(fileObject)
                    
            # Run the Simulation
            self.run()
 

    def run(self):
        self.slice_source_all_slices()  # Sets slice array, num_slices and a_values
        self.create_PSFs_for_source3D()  # Sets PSFs
        self.Get_detector_images_conv()  # Sets Detector_images_2D_noisefree = None
        self.Get_summed_detector_image_conv()  # Sets Detector_image_noisey and noisefree
        
#===============================SECTION: Simulation methods===========================================
    def PSF_raycast_from_mask_keep_resolution(self, a):
        """
        Generates the PSF on the detector plane by casting rays from ideal point source at [0,0,0] through holes in the mask at given a value
        Uses the same resolution as the detector array and allows variable shape of shadow array

        Advantage: - Guaranteed to hit every hole in mask
                   - Very fast: only loops through mask indices where value = 1
                   - Aperture projection represented properly on edges of array (no hits outside of shadow)
        Disadvantage: - Requires Cropping / resizing to detector array
                      - Weird scaling effect to fill the shadow array resolution (eg 121x121 mask -> 512 x 512 shadow)

        Note: a value is variable so that PSF can be generated for each slice of source array
        """
        
        # Find magnification ratio
        m = (a + self.b) / a

        # Calculate size of the shadow in mm, then convert to pixels using the detector resolution
        shadow_size_mm = self.Mask.size_mm * m
        shadow_size_px = int(np.round(shadow_size_mm / (self.Detector.resolution_mm)))

        # Initialize shadow array
        Shadow_array = np.zeros((shadow_size_px, shadow_size_px))

        # Find projected aperture radius in pixels
        r = int(np.round(self.Mask.aperture_radius_mm * m / self.Detector.resolution_mm))

        # Loop over all nonzero pixels of mask and cast ray through each to the shadow
        nonzero_entries = np.array(np.nonzero(self.Mask.mask_array)).astype(int)
        
        for (i, j) in zip(*nonzero_entries):
            mask_pixel_pos = np.array([-self.Mask.size_mm / 2 + i*self.Mask.pixel_mm + self.Mask.pixel_mm/2,
                                       -self.Mask.size_mm / 2 + j*self.Mask.pixel_mm + self.Mask.pixel_mm/2,
                                       a])

            # Find corresponding distance in mm on shadow array by scaling it with magnification
            shadow_pixel_pos = mask_pixel_pos * m

            # Convert shadow pixel position into index values in shadow array.
            # Indexing starts at [0,0] and distance values in mm at the center
            i_shadow = int(shadow_size_px/2) + int(np.round((shadow_pixel_pos[0]/shadow_size_mm) * shadow_size_px))
            j_shadow = int(shadow_size_px/2) + int(np.round((shadow_pixel_pos[1]/shadow_size_mm) * shadow_size_px))

            # Use OpenCV to draw the circle of projected radius
            cv2.circle(Shadow_array, (i_shadow, j_shadow), r, color=1, thickness=-1)
            
        print("PSF created at distance ", a, " with shape ", Shadow_array.shape)

        return Shadow_array
            
    def slice_source_all_slices(self):
        """
        Slices the source along z direction and stores the slices and their corresponding distances from mask
        """
        
        # Cull any sparse elements in source array
        nonzeros = np.nonzero(self.Source.source_array)
        
        # z values of each slice of array in pixels
        z_vals = np.unique(nonzeros[2])

        # Set num slices
        self.num_slices = len(z_vals)
        self.a_values = np.zeros((self.num_slices))

        # Initialize slices array
        self.slices = np.zeros((self.Source.size_px, self.Source.size_px, self.num_slices))
        
        # Fill slices array
        spanstart = self.a - self.Source.size_z_mm/2
        for i in range(self.num_slices):
            self.slices[:, :, i] = self.Source.source_array[:, :, z_vals[i]]
            self.a_values[i] = z_vals[i] * self.Source.resolution_z + spanstart
        

    def create_PSFs_for_source3D(self):
        """
        It generates a PSF for every slice of the object along z axis by calling the PSF method.
        """
        
        self.PSFs = []  # Initialize empty python list to store different shaped PSFs as numpy arrays

        for k in range(self.num_slices):
            if self.conv_mode == "default":
                # Append the PSF and dont bother cropping to 512x512. convolution will be more accurate but may take eternity
                self.PSFs.append(self.PSF_raycast_from_mask_keep_resolution(self.a_values[k]))
                
            elif self.conv_mode == "quick":
                # Append PSF but first crop or pad to 512x512 convolution will be very fast
                self.PSFs.append(HelperFuncs.pad_or_crop(self.PSF_raycast_from_mask_keep_resolution(self.a_values[k]), 512))  
                
            print("PSF shape appended:", self.PSFs[k].shape)  # good for keeping track of PSF size as they are made
            


    def Get_detector_images_conv(self):
        """
        Returns an array of detector images for the given array of PSFs and object slices by equation
            p = f * h where p is forward projection, f is object array, h is PSF array, and * is convolution
        """
        #First doll out the photons based on the inverse square law
        
        # Calculate proportion of photons each pixel gets from each slice, and each slice from the total
        scaled_a_values = np.copy(self.a_values).astype(np.float64)
        
        for i in range(self.num_slices):
            slice_pixel_count = np.sum(self.slices[:, :, i]).astype(int)
            scaled_a_values[i] = (slice_pixel_count / scaled_a_values[i]**2) # This made sense a few months ago
            
            self.photon_alottment = np.round(scaled_a_values * self.Photon_count_total / np.sum(scaled_a_values), 1)  # How much photon count to allot for each slice
            
        # Initialize empty arrays to store projections
        detector_images_raw = np.zeros((self.Detector.size_px, self.Detector.size_px, self.num_slices))
        normalized_detector_images = np.zeros((self.Detector.size_px, self.Detector.size_px, self.num_slices))
        
        self.Detector_images_2D_noisefree = np.zeros((self.Detector.size_px, self.Detector.size_px, self.num_slices))

        # Use the PSF for each slice and do the convolution
        for i in range(self.num_slices):
            PSF_a = self.PSFs[i]
            detector_images_raw[:, :, i] = HelperFuncs.custom_tf_conv2d(self.slices[:, :, i], PSF_a)
            
            # Normalize the detector image so that we can apply photons
            normalized_detector_images[:, :, i] = detector_images_raw[:,:,i] / np.sum(detector_images_raw[:, :, i])

        for i in range(self.num_slices):
            # Apply photons from each slice. if you sum up this whole array you should get the total photon count back
            self.Detector_images_2D_noisefree[:, :,i] = normalized_detector_images[:, :, i] * self.photon_alottment[i]


    def Get_summed_detector_image_conv(self):
        """
        Returns the forward projection of the entire 3D object and simulates poisson noise
        """
        self.Detector_image_noisefree = np.zeros((self.Detector.size_px, self.Detector.size_px)).astype(np.float64) # these need to be float64 or np.poisson will freak out
        self.Detector_image_noisey = np.zeros((self.Detector.size_px, self.Detector.size_px)).astype(np.float64)
        
        # Sum the detector image stack into a single multiplexed image
        for i in range(self.Detector_images_2D_noisefree.shape[2]):
            self.Detector_image_noisefree += self.Detector_images_2D_noisefree[:, :, i]

        # Replace entire image with poisson noise where each pixel is sampled from a poisson distribution with pixel value as the mean
        self.Detector_image_noisey = np.random.poisson(self.Detector_image_noisefree)


#=============================================SECTION: Visualization methods============================================

    def plot_slices(self):
        """
        Just plots the slices of the source. Beware if the source has a lot of slices, it will plot them all!
        """
        cols = self.num_slices
        rows = 1
        fig, ax = plt.subplots(rows, cols, figsize=(cols*5, rows*5.5))

        for i in range(self.num_slices):

            ax[i].imshow(self.slices[:, :, i], cmap="gray")
            titlestr = "Slice " + str(i) + " at a = " + str(np.around(self.a_values[i], 2)) + " mm"
            ax[i].set_title(titlestr)
            
            
    def plot_source3D(self, input_source, vis_mode="points", swap_points=True, elevation=20, azimuth=120):
        """
        Vis modes: "points": looks ugly but fast. "voxels": looks great but takes forever

        for viewing .STL files made with the method above: Using azimuth = 255 and swap_points = False. No idea why.
        """

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        ax.set_xlim([0, 256])
        ax.set_ylim([0, 256])
        ax.set_zlim([0, 256])

        zlabelstr = 'z axis: ' + str(256) + \
            ' pixels, ' + str(57.75) + 'mm'
        xlabelstr = 'x axis: ' + str(256) + \
            ' pixels, ' + str(57.75) + 'mm'
        ylabelstr = 'y axis: ' + str(256) + \
            ' pixels, ' + str(57.75) + 'mm'
        ax.set_xlabel(zlabelstr)
        ax.set_ylabel(ylabelstr)
        ax.set_zlabel(xlabelstr)

        if vis_mode == "points":
            nonzeros = np.nonzero(input_source.source_array)

            if swap_points == True:
                # z and x are purposefully switched
                ax.scatter(nonzeros[2], nonzeros[1], nonzeros[0])
            elif swap_points == False:
                ax.scatter(nonzeros[0], nonzeros[1], nonzeros[2])

        elif vis_mode == "voxels":
            ax.voxels(np.flip(np.swapaxes(input_source.source_array, 0, 2), 2))

        ax.view_init(elev=elevation, azim=azimuth)
        plt.show()

        
    def plot_detector_image(self):
        fig, ax = plt.subplots(1, 1, figsize = (1*5,1*5.5))
        im = plt.imshow(self.Detector_image_noisey)
        figtitlestr = "Detector image for: " + self.simulation_name + "\n with photon count = " + str(self.Photon_count_total)
        fig.suptitle(figtitlestr, fontsize=16)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    def plot_detector_image_noisefree(self):
        fig, ax = plt.subplots(1, 1, figsize = (1*5,1*5.5))
        im = plt.imshow(self.Detector_image_noisefree)
        figtitlestr = "Noisefree Detector image for: " + self.simulation_name + "\n with photon count = " + str(self.Photon_count_total)
        fig.suptitle(figtitlestr, fontsize=16)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    def plot_detector_image_slices(self):
        
        rows = int(self.num_slices/5)
        cols = 3
        
        fig, ax = plt.subplots(rows, cols, figsize = (cols*7,rows*7))
        figtitlestr = "Detector images of slices for " + self.simulation_name
        fig.suptitle(figtitlestr, fontsize=16)
        
        
        for row in range(self.num_slices):
            ind = row*5
            
            im = ax[row, 0].imshow(self.slices[:,:,ind], cmap = "gray")
            titlestr = "Slice " + str(ind) + " at " + str(np.around(self.a_values[ind], 2)) + " mm"
            ax[row, 0].set_title(titlestr)
            fig.colorbar(im, ax=ax[row,0], fraction=0.046, pad=0.04)
            
            im = ax[row, 2].imshow(self.Detector_images_2D_noisefree[:,:,ind])
            titlestr = "Detector Image"
            ax[row, 2].set_title(titlestr)
            fig.colorbar(im, ax=ax[row,2], fraction=0.046, pad=0.04)
            
            im = ax[row, 1].imshow(self.PSFs[ind])
            titlestr = "PSF at "+ str(np.around(self.a_values[ind], 2)) + " mm"
            ax[row, 1].set_title(titlestr)
            fig.colorbar(im, ax=ax[row,1], fraction=0.046, pad=0.04)
            
            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        