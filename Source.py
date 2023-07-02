"""
This class creates the source object (3D binary array) used in the CAI simulation.

Last update: 06.09.2022
@author: Saverio Pietrantonio
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw 
import trimesh

class SourceCalculator():
    def __init__(self, source_type, size_px, size_xy_mm, size_z_mm,
                constrain_to_bounds = None, z_scaling = None,
                sphere3D_radius_mm = None, sphere3D_center = None, 
                multisphere3D_num_spheres = None, multisphere3D_radii = None, multisphere3D_sphere_coords = None,
                rect3D_half_side_lengths_xyz_mm = None, rect3D_center = None,
                multirect3D_num_rects = None, multirect3D_side_lengths = None, multirect3D_rect_coords = None,
                letters3D_string = None, letters3D_z_depths_mm = None, letters3D_z_thicknesses_mm = None,
                stl_string = None, 
                point_source_multiple_num_sources = None, point_source_multiple_coords = None,
                point_source_multiple_coords_mm = None,
                existing_source_array = None):
        """
        Source array is created automatically when object instance is initialized. Each source type has its own required
            parameters set, which are separated by each line above. eg. "sphere3D" only needs sphere3D_radius_mm and sphere3D_center
        
        string source_type: what kind of source shape. Supported ones are below:
            "psource2D", 'psource2D_offset', 'psource3D', "sphere3D", "multisphere3D", "rect3d", "multirect3D", "letters3D", "stl", "mu_tubes"
        int size_px: shape of 3D source array. 256 will give a 256x256x256 numpy array 
        float size_xy_mm: true size of the source array "in real life" in the xy direction (facing the detector plane)
        float size_z_mm: true size of source array in z direction
        
        bool constrain_to_bounds: if true, some spherical or rectangular sources will be shifted to fit within bounds of the source array
            ie, they won't be cut off and you wont see half a sphere
        bool z_scaling: if true, retains the shape of spherical source to always look like a sphere.
            some spherical or rectangular sources will look like ellipsoids if the size_z_mm and size_xy_mm are different,
            since the array shape is always a cube but the pixel resolution in each direction can change.
            Eg: sphere with radius 6 mm: will be N pixels radius in xy and M pixels radius in z, making it look funny.
            
        3D array existing_source_array: only add this if you already have a 3D numpy array to use and just need a source object
        """
        self.size_px = size_px
        self.size_xy_mm = size_xy_mm
        self.size_z_mm = size_z_mm
        self.resolution_xy = self.size_xy_mm / self.size_px  # mm/pixel in x and y directions
        self.resolution_z = self.size_z_mm / self.size_px # mm/pixel in z directions
        self.center = int(self.size_px/2)  # the center pixel
        
        self.source_type = source_type

        
        self.constrain_to_bounds = constrain_to_bounds
        self.z_scaling = z_scaling
        
        # params used for sphere3D
        self.sphere3D_radius_mm = sphere3D_radius_mm
        self.sphere3D_center = sphere3D_center 
        
        # params used for multisphere3D
        self.multisphere3D_num_spheres = multisphere3D_num_spheres
        self.multisphere3D_radii = multisphere3D_radii
        self.multisphere3D_sphere_coords = multisphere3D_sphere_coords
        
        # params used for rect3D
        self.rect3D_half_side_lengths_xyz_mm = rect3D_half_side_lengths_xyz_mm
        self.rect3D_center = rect3D_center
        
        # params used for multirect3D
        self.multirect3D_num_rects = multirect3D_num_rects
        self.multirect3D_side_lengths = multirect3D_side_lengths
        self.multirect3D_rect_coords = multirect3D_rect_coords
        
        # params used for letters3D
        self.letters3D_string = letters3D_string
        self.letters3D_z_depths_mm = letters3D_z_depths_mm
        self.letters3D_z_thicknesses_mm = letters3D_z_thicknesses_mm
        
        # params used for stl
        self.stl_string = stl_string
        
        # params used for multiple point sources
        self.point_source_multiple_num_sources = point_source_multiple_num_sources
        self.point_source_multiple_coords = point_source_multiple_coords
        self.point_source_multiple_coords_mm = point_source_multiple_coords_mm
        
        # initialize empty source array
        self.source_array = None
        
        if existing_source_array == None:
            self.source_array = np.zeros((self.size_px, self.size_px, self.size_px), dtype=np.float32)  # initialize empty array. to be set by methods
            self.set_source_array()  # This is where the source array is set
        else:
            print("creating source from given array")
            self.source_array = existing_source_array
            
    
    def set_source_array(self):
        # called automatically if no existing source array is given
        # Just redirects the creation of the source depending on source_type
        
        
        if self.source_type == 'psource2D':
            print("Creating psource2D")
            self.create_psource2D()
        elif self.source_type == 'psource2D_offset':
            print("Creating psource2D offset")
            self.create_psource2D_offset()
        elif self.source_type == 'psource3D':
            print("Creating psource3D")
            self.create_psource3D()
        elif self.source_type == 'sphere3D':
            self.create_sphere3D()
        elif self.source_type == 'multisphere3D':
            self.create_multisphere3D()
        elif self.source_type == 'rect3D':
            self.create_rect3D()
        elif self.source_type == 'multirect3D':
            self.create_multirect3D()
        elif self.source_type == 'letters3D':
            self.create_letters3D()
        elif self.source_type == 'stl':
            self.stl_to_source()
        elif self.source_type == "mu_tubes":
            self.create_Mu2006_tubes()
        elif self.source_type == "point_source_multiple":
            print("Creating multiple point source object")
            self.create_psource_multiple()
            
            
    # This section contains all the functions for creating the different source objects

        
    def create_psource2D(self):
        source_array = np.zeros((self.size_px,self.size_px))
        source_array[self.center - 1 : self.center + 1 , self.center - 1 : self.center + 1] = 1
        return source_array
     
    def create_psource2D_offset(self):
        source_array = np.zeros((self.size_px, self.size_px))
        source_array[0:1, 0:1] = 1
        return source_array
        
    def create_psource3D(self, source_array, radius = 1):
        source_array = np.zeros((self.size_px, self.size_px, self.size_px))
        source_array[self.center-radius : self.center+radius ,
                     self.center-radius : self.center+radius, 
                     self.center-radius : self.center+radius] = 1
        return source_array
    
    def create_psource_multiple(self):
        for i in range(self.point_source_multiple_num_sources):
            self.source_array[self.point_source_multiple_coords[i,0],
                         self.point_source_multiple_coords[i,1],
                         self.point_source_multiple_coords[i,2]] = 1
                
    def create_sphere3D(self): 
        """
        Creates a spherical source
        """
        
        # Find radius in xy and z in pixels
        r_xy = int(np.round(self.sphere3D_radius_mm / self.resolution_xy))
        r_z = int(np.round(self.sphere3D_radius_mm / self.resolution_z))
        min_radius = np.amin([r_xy, r_z])
        max_radius = np.amax([r_xy, r_z])

        if self.constrain_to_bounds == True:
            # This makes sure the entire sphere is always in the boundary of cube. changes sphere center if not
            distx = self.size_px - self.sphere3D_center[0]  # distance from sphere center to edge of box in px
            disty = self.size_px - self.sphere3D_center[1]
            distz = self.size_px - self.sphere3D_center[2]
            if distx < r_xy:  # Sphere will overlap in x
                self.sphere3D_center[0] = self.source_array.shape[0] - r_xy # Set center of circle on the edge of the cube defined by the edge - radius
                
            if disty < r_xy:  # Sphere will overlap in y
                self.sphere3D_center[1] = self.source_array.shape[1] - r_xy
                
            if distz < r_z:  # Sphere will overlap in z
                self.sphere3D_center[2] = self.source_array.shape[2] - r_z
                
            # Now check for minimum bounds
            if self.sphere3D_center[0] < r_xy:
                self.sphere3D_center[0] = r_xy
            if self.sphere3D_center[1] < r_xy:
                self.sphere3D_center[1] = r_xy
            if self.sphere3D_center[2] < r_z:
                self.sphere3D_center[2] = r_z
        
        print("creating sphere with px radii:", r_xy, r_z, " at ", self.sphere3D_center)
        
        # Creates a bounding box for the sphere
        x_square_min = self.sphere3D_center[0]-r_xy
        x_square_max = self.sphere3D_center[0]+r_xy+1
        y_square_min = self.sphere3D_center[1]-r_xy
        y_square_max = self.sphere3D_center[1]+r_xy+1
        z_square_min = self.sphere3D_center[2]-r_z
        z_square_max = self.sphere3D_center[2]+r_z+1

        if x_square_min < 0:
            x_square_min = 0
        if y_square_min < 0:
            y_square_min = 0
        if z_square_min < 0:
            z_square_min = 0

        if x_square_max > self.size_px:
            x_square_max = self.size_px
        if y_square_max > self.size_px:
            y_square_max = self.size_px
        if z_square_max > self.size_px:
            z_square_max = self.size_px

        # Iterates over each pixel in the bounding box, assigning 1 if falls within radius of sphere
        for p in range(x_square_min , x_square_max):
            for q in range(y_square_min , y_square_max):
                for u in range(z_square_min, z_square_max):
                    if self.z_scaling == True:
                        ratio_r = max_radius / min_radius  # This is used to scale the ellipsoid
                        
                        if (p - self.sphere3D_center[0]) ** 2 + (q - self.sphere3D_center[1]) ** 2 + ((u - self.sphere3D_center[2])**2)/ratio_r < min_radius ** 2:
                            self.source_array[p,q,u] = 1

                    elif self.z_scaling == False:
                        if (p - self.sphere3D_center[0]) ** 2 + (q - self.sphere3D_center[1]) ** 2 + ((u - self.sphere3D_center[2])**2) < min_radius ** 2:
                            self.source_array[p,q,u] = 1
                            
                    elif self.z_scaling == None:
                        print("error: you didnt put in z scaling parameter in source creation create_sphere3D()")
                        

    def create_multisphere3D(self):
        """
        Creates multiple spheres in one 3D array.
        Works by reassigning the values of sphere radius and center and calling self.create_sphere3D for each new sphere()
        int num_spheres: how many spheres to show in the plot?
        1D np array radii: radius in mm of each sphere
        2D np array: Nx3 array giving x,y,z center of each sphere
        """
        
        if  self.multisphere3D_num_spheres != self.multisphere3D_radii.shape[0] or self.multisphere3D_num_spheres != self.multisphere3D_sphere_coords.shape[0]:
            raise ValueError('multisphere3D: radii and coords must match number of spheres')
            return
        
        for i in range(self.multisphere3D_num_spheres):
            # First modifies the radius and center of Sphere3D
            self.sphere3D_center = self.multisphere3D_sphere_coords[i]
            self.sphere3D_radius_mm = self.multisphere3D_radii[i]  
            
            self.create_sphere3D()  # Sphere3D function doesnt care if a sphere already exists or not, so just call it again
        
    def create_rect3D(self): 
        """
        Same as the sphere3D but just makes rectangles
        """
        #Turn mm into pixels: assume length_mm is half of the length of the rectangle in that dimension            
        length_x = int(np.round(self.rect3D_half_side_lengths_xyz_mm[0] / self.resolution_xy))
        length_y = int(np.round(self.rect3D_half_side_lengths_xyz_mm[1] / self.resolution_xy))
        length_z = int(np.round(self.rect3D_half_side_lengths_xyz_mm[2] / self.resolution_z))
        
        if self.constrain_to_bounds == True:
            # This makes sure the entire sphere is always in the boundary of cube
            distx = self.size_px - self.rect3D_center[0]  
            disty = self.size_px - self.rect3D_center[1]
            distz = self.size_px - self.rect3D_center[2]
            
            if distx < length_x:  
                self.rect3D_center[0] = self.size_px - length_x
            if disty < length_y:  
                self.rect3D_center[1] = self.size_px - length_y 
            if distz < length_z: 
                self.rect3D_center[2] = self.size_px - length_z
            if self.rect3D_center[0] < length_x:
                self.rect3D_center[0] = length_x
            if self.rect3D_center[1] < length_y:
                self.rect3D_center[1] = length_y 
            if self.rect3D_center[2] < length_z:
                self.rect3D_center[2] = length_z

        x_square_min = self.rect3D_center[0]-length_x
        x_square_max = self.rect3D_center[0]+length_x+1
        y_square_min = self.rect3D_center[1]-length_y
        y_square_max = self.rect3D_center[1]+length_y+1
        z_square_min = self.rect3D_center[2]-length_z
        z_square_max = self.rect3D_center[2]+length_z+1

        if x_square_min < 0:
            x_square_min = 0
        if y_square_min < 0:
            y_square_min = 0
        if z_square_min < 0:
            z_square_min = 0

        if x_square_max > self.size_px:
            x_square_max = self.size_px
        if y_square_max > self.size_px:
            y_square_max = self.size_px
        if z_square_max > self.size_px:
            z_square_max = self.size_px

        self.source_array[x_square_min : x_square_max, y_square_min : y_square_max, z_square_min : z_square_max] = 1

    def create_multirect3D(self):
        """ 
        make lots of retangles
        """
        if self.multirect3D_num_rects != self.multirect3D_side_lengths.shape[0] or self.multirect3D_num_rects != self.multirect3D_rect_coords.shape[0]:
            print("side lengths or centers much match number of rectangles")
            return
        
        for i in range(self.multirect3D_num_rects):
            self.rect3D_half_side_lengths_xyz_mm = self.multirect3D_side_lengths[i]
            self.rect3D_center = self.multirect3D_rect_coords[i]
            self.create_rect3D()  
        
    def create_letter2D(self, letter):
        """
        Supporting function for creating letters3D. Cannot be called alone. 
        
        This takes in a letter string like 'H' and turns it into a 2D binary image
        letter: string of 1 capital letter
        z_depth_mm: how far into the image should the midpoint of the letter be
        z_thickness_mm: how thick should the letter be in z
        """
        if self.size_px != 256:
            raise ValueError('create_letter2D only works if source size is 256 x 256')
        
        img = Image.new('1', (self.size_px, self.size_px), "black")  # Create a binary image using pillow

        # Get a drawing handle and do pillow stuff. 
        # Values like fontsize and text size are just found by experimenting, no rhyme nor reason
        draw = ImageDraw.Draw(img)
        fontsize = 340
        font = ImageFont.truetype("arial.ttf", fontsize, encoding="unic")
        draw.textsize(letter, font=font)
        draw.text((128, 132), letter,  font = font, fill="white", anchor="mm", align ="center", stroke_width = 1) 
        na = np.array(img).astype(int) # Convert PIL Image to Numpy array for processing
        return np.rot90(na,3)
    
    def create_letters3D(self):
        """
        Creates a 3D array of letters that are stacked one behind the other
        
        string: string of capital letter eg 'TOBIISAWESOME'
        z_depth_mm: np array of floats with length = len(letters), eg [14.3, 20.2, 50.0]
            ---> how far into the z direction should the midpoint of the letters be
            If left None, will automatically fit the letters perfectly
        z_thicknesses_mm: np array of floats with length = len(letters), eg [2.4, 4.5, 6.7] 
            ---> how thick each letter is (in z direction)
            If left None, will = z_depth_mm / 2
        """
        z_thicknesses_px = np.round(self.letters3D_z_thicknesses_mm / self.resolution_z).astype(int)   # Convert values to pixels in z
        z_depths_px = np.round(self.size_px * (self.letters3D_z_depths_mm / self.size_z_mm)).astype(int)
        
        for i in range(len(self.letters3D_string)):
            # 3D array of letters is created by creating multiple 2D letter slices and stacking them on top of each other
            self.source_array[:,:, z_depths_px[i] : z_depths_px[i] + z_thicknesses_px[i]] = np.expand_dims(self.create_letter2D(self.letters3D_string[i]), axis = 2)
                                   
        
    def create_tubes(self, num_tubes, diameter_mm, length_mm, start_points, directions_2d):
        """
        In some literature they like to use radioactive tube phantoms filled with Tc99 to replicate extended sources. 
        This will create tubes to whatever specification you want
        
        int num_tubes: number of tubes
        directions_2d: an 2 x n array of 2d vectors which give the direction of each tube from starting point        
        float length_mm: length of tubes
        float diameter_mm: diameter of tubes
        start_points: pixel index where to start the tube
        
        Creates tubes by calling sphere3D a bunch of times in a row along a straight line. Kind of like using a ballpoint pen on paper. 
        This gives the tubes a nice rounded edge
        """
        self.sphere3D_radius_mm = diameter_mm/2
        

        for i in range(num_tubes):
            current_pt = start_points[i]
            current_length = 0
            
            euc_distance = np.sqrt((directions_2d[i,0]*self.resolution_xy)**2 + (directions_2d[i,1]*self.resolution_xy)**2)
            while current_length <= length_mm:
                # Move sphere center to overlap on top of current point along the line and draw the sphere
                self.sphere3D_center = np.asarray([current_pt[0], current_pt[1], current_pt[2]])
                self.create_sphere3D()
               
                # move to the next point and update the length of the tube
                current_pt[0] += directions_2d[i,0]
                current_pt[1] += directions_2d[i,1]
                current_length += euc_distance

        
    def create_Mu2006_tubes(self):
        """
        tries to recreate the depth experiment in Mu 2006. Makes a '>H' using tubes
        Finds the correct position of tubes and calls self.create_tubes() 5 times to draw the letters
        """
        
        """
        Experimental details
        # use a = 21 cm
        # H at z = 18.4 cm
        # > at z = 22.4 cm
        """
        
        tube_length = 30/2
        directions = np.zeros((5,2)) # 2 for each tube for x,y and 5 for 5 tubes
        start_points = np.zeros((5,3)) # 3 for each coordinate, and 5 for 5 tubes
        
        
        # Find starting point in z
        H_length_mm = 210 - 184
        v_length_mm = 210 - 224
        
        H_length_px = H_length_mm / self.resolution_z
        v_length_px = v_length_mm / self.resolution_z
        
        array_center = np.asarray([self.center,self.center,self.center])
        
        H_start_z = array_center[2] - H_length_px
        v_start_z = array_center[2] - v_length_px
        
        start_points[0:3,2] = H_start_z
        start_points[3:5,2] = v_start_z
        
        # Find starting point in x,y
            # H
                # left leg of H
                     # x
        start_points[0,0] = array_center[0] - tube_length / self.resolution_xy
        directions[0,0] = 0 # const x
                    # y
        start_points[0,1] = array_center[1] - tube_length / self.resolution_xy  
        directions[0,1] = 1 # positive y
                
                # middle arch of H
                    # x
        start_points[1,0] = array_center[0] - tube_length / self.resolution_xy
        directions[1,0] = 1  # positive x
        
                    # y
        start_points[1,1] = array_center[1]
        directions[1,1] = 0 # constant in y   
        
                 # right leg of H
                     # x
        start_points[2,0] = array_center[0] + tube_length / self.resolution_xy
        directions[2,0] = 0 # const x
                    # y
        start_points[2,1] = array_center[1] - tube_length / self.resolution_xy  
        directions[2,1] = 1 # positive y        
        
        # >
                # top of >
                     # x - top left to center
        start_points[3,0] = array_center[0] - tube_length / self.resolution_xy - 30 + 60
        directions[3,0] = 1 # pos x
                    # y
        start_points[3,1] = array_center[1] + tube_length / self.resolution_xy  + 30
        directions[3,1] = -1 # neg y
                # bottom of >
                     # x - bottom left to center
        start_points[4,0] = array_center[0] - tube_length / self.resolution_xy - 30 + 60
        directions[4,0] = 1 # pos x
                    # y
        start_points[4,1] = array_center[1] - tube_length / self.resolution_xy - 30
        directions[4,1] = 1 # pos y        
        
        
        self.create_tubes(num_tubes = 5, diameter_mm = 4.0, length_mm = 15,
                          start_points = start_points.astype(int), directions_2d = directions)
        
    def stl_to_source(self):
        """
         Brilliantly turns a stl file from eg. Blender into a 3D binary source array. 
         This took me about 30 hours to do and we never used it. Such is life.
        """
        # stl vertex coordinates represents spatial location in mm. need to convert these to pixels
        stl_arr = trimesh.load_mesh(self.stl_string, enable_post_processing=True, solid=True) # Import Objects
        vertices = stl_arr.vertices
        #Scaling the STL file to fit in the array and converting mm to pixels
        max_x = np.amax(vertices[:,0])  # max size in x
        max_y = np.amax(vertices[:,1])
        max_z = np.amax(vertices[:,2])
        min_x = np.amin(vertices[:,0])
        min_y = np.amin(vertices[:,1])
        min_z = np.amin(vertices[:,2])

        stretch_max_x = (self.size_xy_mm/2) / (max_x)
        stretch_max_y = (self.size_xy_mm/2) / (max_y)
        stretch_max_z = (self.size_z_mm/2) / (max_z)

        stretch_min_x = (-self.size_xy_mm/2) / (min_x)
        stretch_min_y = (-self.size_xy_mm/2) / (min_y)
        stretch_min_z = (-self.size_z_mm/2) / (min_z)

        stretch_x = np.amin([stretch_max_x, stretch_min_x])
        stretch_y = np.amin([stretch_max_y, stretch_min_y])
        stretch_z = np.amin([stretch_max_z, stretch_min_z])

        # Stretch them to keep proportions, then shift them to get rid of negatives
        vertices[:,0] = vertices[:,0]*stretch_x + self.size_xy_mm/2
        vertices[:,1] = vertices[:,1]*stretch_y + self.size_xy_mm/2
        vertices[:,2] = vertices[:,2]*stretch_z + self.size_z_mm/2

        resolution_xy = self.size_xy_mm / (self.size_px-1)
        resolution_z = self.size_z_mm / (self.size_px-1)

        pix_size_xy = 1/resolution_xy
        pix_size_z = 1/resolution_z

        vertices[:,0] *= pix_size_xy
        vertices[:,1] *= pix_size_xy
        vertices[:,2] *= pix_size_z

        vertices = vertices.astype(int)

        voxel_array = np.zeros((self.size_px, self.size_px, self.size_px))

        for x, y, z in vertices: voxel_array[x, y, z] = 1 # assign vertices to voxels
        self.source_array = voxel_array 
        
    def plot_source_slices_3D(self,  slicenum, elevation=20, azimuth=120):
        """ 
        Used to visualize slices of the source using matplotlib voxels. slicenum = slice to highlight
        """
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        
        voxelarray = np.copy(self.source_array)


        voxelarray1 = np.copy(voxelarray)  # main object
        voxelarray2 = np.copy(voxelarray) # slice

        voxelarray1[slicenum,:,:] = 0  # delete slice from first
        voxelarray1[0:slicenum,:,:] = 0 # delete everything before the slice

        #delete all but slice from 2nd
        voxelarray2[0:slicenum, :, :] = 0
        voxelarray2[slicenum+1:, :, :] = 0

        ax.voxels(voxelarray1)
        ax.voxels(voxelarray2, facecolors='red')
        
        
        ax.view_init(elev=elevation, azim=azimuth)
        plt.show()

                
                
