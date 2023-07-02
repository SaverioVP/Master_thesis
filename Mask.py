"""
https://www.imdb.com/title/tt0110475/
"""

import numpy as np

class MuraMaskCalculator:
    def __init__(self, L: int, size_mm = 22.08, thickness = 1.0, aperture_shape = "circle", aperture_radius_mm = 0.170):
        """
        Creates a Coded Aperture mask object. the main mask array is contained in self.mask_array but 
            the simulation needs other info as well which is kept here. Default values are from the Rozhkov setup

        """
        self.L = L  # Rank of the mask
        self.mask_array = self.create(L)  # Creates the mask array
        
        # Mask Properties
        self.size_px = self.mask_array.shape[0]
        self.size_mm = size_mm  # height of the mask
        self.t = thickness # thickness of the mask. Unused
        self.aperture_shape = aperture_shape  # aperture shape of the mask. Unused
        self.aperture_radius_mm = aperture_radius_mm  # aperture radius. assumes circular apertures
        
        #Calculated vals
        self.pixel_mm = self.size_mm / self.size_px

            
    def create(self, L: int):
        # Creates the MURA NTHT mask based on quadratic residues with rank L. 
        # This is just Tobi'S code slightly modified for readability
        
        quadratic_res = np.array([])
        for x in range(int(np.ceil(L / 2))):
            quadratic_res = np.append(quadratic_res, np.remainder(x ** 2, L))
        quadratic_res = np.unique(quadratic_res)
        A = np.zeros((L, L))
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                x_is_quadratic_residue = np.any(np.in1d(quadratic_res, x))
                y_is_quadratic_residue = np.any(np.in1d(quadratic_res, y))
                if x == 0:
                    A[x, y] = 0
                elif y == 0 and x != 0:
                    A[x, y] = 1
                elif x_is_quadratic_residue and y_is_quadratic_residue:
                    A[x, y] = 1
                elif not x_is_quadratic_residue and not y_is_quadratic_residue:
                    A[x, y] = 1
                else:
                    A[x, y] = 0
        ntht = np.zeros((A.shape[0] * 2, A.shape[1] * 2))
        ntht[::2, ::2] = A
        ntht = np.rot90(ntht)
        ntht = np.tile(ntht, (2, 2))
        ntht = ntht[1:-2, 2:-1]
        return ntht
    