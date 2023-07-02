"""
This class handles some miscellaneous functions which are are used often

Last update: 06.09.2022
@author: Saverio Pietrantonio
"""
import tensorflow as tf
import numpy as np



class HelperFuncs():
    def __init__(self):
        pass
        
    
    def custom_tf_conv2d(_input, _filter):
        """
        custom wrapper for tf.nn.conv2d which is a pain in the ass to use alone. This will accept 
        the raw input array and filter and return their convolution, in the same dimensions as given.
        
        If the input is 256x256 and filter is 512x512, which is used in the simulation, then call Tobi's "hack"
        to get a very fast convolution

        """

        if _input.shape == (256,256) and _filter.shape == (512,512):
            # Cast input and filter to proper tensor shapes for use by tensorflow
            _input = tf.reshape(_input, [1, _input.shape[0], _input.shape[1], 1]).numpy() 
            _filter = tf.reshape(_filter, [1,_filter.shape[0], _filter.shape[1],1]).numpy() 
            
            # Do Tobi's evil hack
            z = tf.transpose(_input, [1, 2, 3, 0])
            y = tf.nn.conv2d(_filter, z, strides=1, padding="VALID")
            y2 = tf.transpose(y, [3, 1, 2, 0])
            y3 = tf.slice(y2, (0, 0, 0, 0), (-1, 256, 256, -1))
            y4 = tf.image.flip_up_down(tf.image.flip_left_right(y3))
            
            # Return the convolution and return back to 2D using np squeeze
            return np.squeeze(y4)
        
        else:
            
            # print("doing long convolution...")   # Uncomment this if you want to measure the time it takes
            # start_time = time.process_time()
            _input = tf.reshape(_input, [1, _input.shape[0], _input.shape[1], 1]).numpy() 
            _filter = tf.reshape(_filter, [_filter.shape[0], _filter.shape[1],1,1]).numpy() 
            # final_time = time.process_time() - start_time
            # print("long convolution completed in ", final_time, " seconds")
            
            # Simply call tf.nn.conv2d and return 2D output with squeeze
            return np.squeeze(tf.nn.conv2d(_input, _filter, strides=1, padding="SAME"))
        
        
    
    
    def pad_or_crop(array, size):
        """
        This function takes a 2D np array and an int array shape, eg. 512 (for 512x512) and will pad or crop it to that size-
        Array must be square, ie. NxN
        
        """
        
        if size == array.shape[0]: # array is same size. do nothing
            return array
        
        # Calculate how much need to pad or crop
        size_diff = size - array.shape[0]  # ASSUMES ARRAY IS SQUARE!
        pad_left_up = np.abs(np.floor(size_diff/2)).astype(int)
        pad_right_down = np.abs(np.ceil(size_diff/2)).astype(int)
    
        if size > array.shape[0]:  # array is smaller than size. Add padding
            padding = ((pad_left_up, pad_right_down),(pad_left_up, pad_right_down))  # up down left right
            new_array = np.pad(array, padding, mode='constant', constant_values = 0)
    
        elif size < array.shape[0]:  # array is larger than size. Crop. 
            new_array = array[array.shape[0]//2 - size//2 : array.shape[0]//2 + size//2,
                              array.shape[1]//2 - size//2 : array.shape[1]//2 + size//2]  # also assumes square
        return new_array
    