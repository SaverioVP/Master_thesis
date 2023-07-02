"""
This class is for mass producing 3D sources to be used if you need a LOT of data.
At the moment can only make multisphere, multirect and letters source objects
Saves sources objects as Pickle files
"""

import numpy as np


import pickle

from Source import SourceCalculator


class Data_Creator:
    def __init__(self, num_each = 5, types = ["multisphere3D", "multirect3D", "letters3D"], 
                 auto_generate = True, auto_save = False, folder = None):
        """
        num_each: how many data files of each type
        types: array of strings of types. "["multisphere3D", "multirect3D", "letters3D"]"
        """
        
        self.num_each = num_each
        self.types = types
        self.num_types = len(types)
        self.auto_save = auto_save
        self.folder = folder
        if auto_generate == True:
            self.generate()
        
    def generate(self):
        for sourcetype in self.types:
            # create N of each source type given
            for i in range(self.num_each):
                if sourcetype == "multisphere3D":
                    print("generating multisphere3D source")
                    Source = self.generate_spheres()
                    
                elif sourcetype == "multirect3D":
                    print("generating multirect3D source")
                    Source = self.generate_rects()
                
                elif sourcetype == "letters3D":
                    print("generating letters3D source")
                    Source = self.generate_letters()
                    
                else:
                    print("source type not recognized. check your input")

                if self.auto_save == True:
                    filename = sourcetype + "_object_" + str(i) + ".pkl"
                    if self.folder != None:
                        folder = self.folder + "/"
                    else:
                        folder = ""
                    path = folder + filename
                    
                    ##Testing this new block
                    with open(path, "wb") as fileObject:
                        pickle.dump(Source, fileObject)
        
    def generate_rects(self, max_num_rects = 8, min_side_length = 3, max_side_length = 10, 
                         source_size_px = 256, souce_size_xy_mm = 57.75, size_z_mm = 57.75):
        multirect3D_num_rects_randomized = np.random.randint(1,max_num_rects +1)
        multirect3D_side_lengths_randomized = np.random.randint(min_side_length,max_side_length, size=(multirect3D_num_rects_randomized , 3)) # in mm
        multirect3D_rect_coords_randomized = np.random.randint(source_size_px , size=(multirect3D_num_rects_randomized , 3))
        
        source_multirect3D = SourceCalculator('multirect3D', size_px= 256, size_xy_mm=57, size_z_mm=57,
                 constrain_to_bounds = True, z_scaling = True,
                 multirect3D_num_rects = multirect3D_num_rects_randomized, 
                 multirect3D_side_lengths = multirect3D_side_lengths_randomized,
                 multirect3D_rect_coords = multirect3D_rect_coords_randomized)
       
        return source_multirect3D
    
    
    def generate_spheres(self, max_num_spheres = 8, min_radius = 3, max_radius = 10, 
                         source_size_px = 256, souce_size_xy_mm = 57, size_z_mm = 57):
        
        multisphere3D_num_spheres_randomized = np.random.randint(1,max_num_spheres+1)  # max 4
        multisphere3D_radii_randomized = np.random.randint(min_radius,max_radius, size=(multisphere3D_num_spheres_randomized))  # in mm
        multisphere3D_sphere_coords_randomized = np.random.randint(source_size_px, size=(multisphere3D_num_spheres_randomized, 3))
        
        source_multisphere3D = SourceCalculator('multisphere3D', size_px= source_size_px, 
                  size_xy_mm=souce_size_xy_mm, size_z_mm=size_z_mm,
                  constrain_to_bounds = True, z_scaling = True,
                  multisphere3D_num_spheres = multisphere3D_num_spheres_randomized, 
                  multisphere3D_radii = multisphere3D_radii_randomized,
                  multisphere3D_sphere_coords = multisphere3D_sphere_coords_randomized)
        
        return source_multisphere3D
        
    def generate_letters(self, max_string_len = 4,
                         source_size_px = 256, source_size_xy_mm= 57, source_size_z_mm = 57):
  
        alphabet= ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
        random_str_len = np.random.randint(2,max_string_len+1)
        
        random_string = np.random.choice(alphabet, size=(1,random_str_len), replace=True)
        #print(random_string)
        source_letters3D = SourceCalculator('letters3D', size_px= source_size_px,
                                            size_xy_mm= source_size_xy_mm, size_z_mm= source_size_z_mm,
                                            letters3D_string  = random_string)
        return source_letters3D
        