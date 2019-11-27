# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array
import matplotlib.image as mpimg
from scipy import signal
import math


############### DISCLAIMER #####################
#### When plotting please do one at a time, otherwise the plots for convolution ####
#### and histogram overlap 


class Convolution:
  def __init__(self):

    NAME = 'NVIDIA CUDA'
    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == NAME:
            devs = platform.get_devices()

    # Set up a command queue:
    self.ctx = cl.Context(devs)
    self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)    
    # Write your naive 2D Convolution kernel here

    self.kernel_code_naive = """

        __kernel void convolution_2D_naive(__global float *in, __global float *mask, __global float *out, 
        const int mask_width, int w, int h) {
            int Col = get_global_id(0); 
            int Row = get_global_id(1); 

            if (Col < w && Row < h){
                float pix_Val = 0;

                int N_start_col = Col - (mask_width/2);
                int N_start_row = Row - (mask_width/2);

                // Get the surrounding box
                for (int j = 0; j < mask_width; ++j){
                    for (int k = 0; k < mask_width; ++k){
                        int cur_Row = N_start_row + j;
                        int cur_Col = N_start_col + k;

                        // Verify the image pixel is valid
                        if (cur_Row > -1 && cur_Row < h && cur_Col > -1 && cur_Col < w){
                            pix_Val += in[cur_Row * w + cur_Col] * mask[j * mask_width + k];
                        }
                    }
                }
                // New pixel value output
                out[Row * w + Col] = pix_Val;
            }
        }
            """
        
    self.prg_naive = cl.Program(self.ctx, self.kernel_code_naive).build()

  def conv2d_gpu(self, A, kernel):
    # INPUTS:
      # A --> matrix    
      # kernel --> the filter

     # Transfer data to device
    self.A_d = cl.array.to_device(self.queue, A)
    self.kernel_d = cl.array.to_device(self.queue, kernel)
    self.output_d = cl.array.empty(self.queue, A.shape, A.dtype)

        
    # Call kernel function
    func_naive = self.prg_naive.convolution_2D_naive

    if (A.shape[0] > A.shape[1]):
        m_A = A.shape[0]
    else:
        m_A = A.shape[1]

    # Measure time
    evt = func_naive(self.queue,(np.int(m_A),np.int(m_A)), None, self.A_d.data, self.kernel_d.data, self.output_d.data, 
    np.int32(kernel.shape[1]), np.int32(A.shape[1]), np.int32(A.shape[0]))
    evt.wait()
    time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time

    output_gpu = self.output_d.get()
    kernel_time = time_

    # Return the result and the kernel execution time
    return output_gpu, kernel_time

  def conv2d_cpu(self, A, kernel):
    # INPUTS:
      # A --> matrix
      # kernel --> the filter

    # Use scipy to compute the 2D convolution
    # between the input matrix and the filter
    start = time.time()
    o_cpu = signal.convolve2d(A, kernel, mode='same')
    serial_time = time.time() - start
    output_cpu = o_cpu
    # Return the result and the serial run time
    return output_cpu, serial_time

class Histogram:
  def __init__(self):

    NAME = 'NVIDIA CUDA'
    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == NAME:
            devs = platform.get_devices()

    # Set up a command queue:
    self.ctx = cl.Context(devs)
    self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)    # Write your naive 2D Convolution kernel here
    # Write your 1D histogram kernel here
    self.kernel_hist = """ 
    __kernel void hist(__global const int *buffer, __global int *hist, const int size)

    {
     int i = get_global_id(0);     

     if (i < size && i >= 0) 
        {
        int bin = buffer[i];
        atomic_add(&hist[bin], 1);
        }
    }
    """

    self.prg_hist = cl.Program(self.ctx, self.kernel_hist).build()

  def hist1d_gpu(self, a, bins=26):
    # INPUTS:
      # a --> 1D array 
      # bins --> the number of bins we want
      #          divide our histogram into
    self.a = a
    self.bins = bins
    a_new = np.floor(self.a.astype(np.float32)/10).astype(np.int32)

    # Transfer data to device
    self.a_d = cl.array.to_device(self.queue, a_new)
    self.out_d = cl.array.zeros(self.queue, (bins,), a_new.dtype)
    
    # Compute block and grid size
    M = len(a)
    
    # Call kernel function
    func_hist = self.prg_hist.hist
    
    # Measure time
    evt = func_hist(self.queue,(int(M),1), None, self.a_d.data, self.out_d.data, np.int32(M))
    evt.wait()
    time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time

    hist_gpu = self.out_d.get()
    kernel_time = time_

    # Return the histogram and the kernel execution time
    return hist_gpu, kernel_time

  def hist1d_cpu(self, a, bins=26):
    # INPUTS:
      # a --> 1D array 
      # bins --> the number of bins we want
      #          divide our histogram into
    self.a = a
    self.bins = bins
    hist_cpu = np.zeros(self.bins, dtype=np.int32)
    M = len(a)

    start_ = time.time()
    for e in range(M):
        bin = int(math.floor(a[e]/10))
        hist_cpu[bin] = hist_cpu[bin] + int(1)
    end_ = time.time()

    serial_time = end_ - start_
    # Return the histogram and the serial execution time
    return hist_cpu, serial_time

if __name__ == '__main__':

  ##########################################################
  #                   CONVOLUTION                          #
  ##########################################################

    cl_times_naive = []
    py_times = []
    matrix_size = []

  # ITERATIVELY:
    for itr in range(1,40):

    # Create input matrix
        A = np.float32(np.random.randint(low=0, high=9, size=(itr*20,itr*25)))
        # A_array = np.array(A, np.float32)
        matrix_size.append(A.shape[0] * A.shape[1])

    # Create kernel filter
        kernel = np.float32(np.random.randint(low=0, high=5, size=(5,5)))
        # kernel_array = np.array(kernel, np.float32)
        kernel_flip = np.rot90(kernel, 2).astype(np.float32)
        # print("KERNEL FLIP:", kernel_flip)

    # Create instance for OpenCL
        module = Convolution()    

        naive_times=[]
        serial_times = []

        # Record times

        for e in range(3):
            # Compute Naive GPU convolution
            cl_output_naive, t = module.conv2d_gpu(A, kernel_flip)
            naive_times.append(t)
            # Compute Tiled GPU Convolution

            # Compute Serial Convolution
            cpu_output, t_cpu = module.conv2d_cpu(A, kernel)
            serial_times.append(t_cpu)
        cl_times_naive.append(np.average(naive_times))
        py_times.append(np.average(serial_times))
        
    # Code Equality
        print("CPU & NAIVE GPU computation:", np.allclose(cl_output_naive, cpu_output))

  # Plot times

    # MAKE_PLOT = True
    # if MAKE_PLOT:
    #     plt.gcf()
    #     plt.plot(matrix_size, py_times,'r', label="Python")
    #     plt.plot(matrix_size, cl_times_naive,'b', label="OpenCL NAIVE")
    #     plt.legend(loc='upper left')
    #     plt.title('2D Convolution')
    #     plt.xlabel('size of array')
    #     plt.ylabel('output coding times(sec)')
    #     plt.gca().set_xlim((min(matrix_size), max(matrix_size)))
    #     plt.gca().set_ylim(0, 0.0075)
    #     plt.savefig('plots_pyopenCL_2D_conv_SMALL.png')  

  ##########################################################
  #                    HISTOGRAM                           #
  ##########################################################

    cl_times_hist = []
    py_times_hist = []
    matrix_size = []

  # ITERATIVELY:
    for itr in range(1,40):
    # Create input array
        a = np.float32(np.random.randint(low=0, high=255, size=(itr*38)))
        matrix_size.append(len(a))
    # Create instance for CUDA
        module = Histogram()    
        
        hist_cpu_t =[]
        hist_gpu_t = []

    # Record times
        for e in range(3):
            # Compute GPU Histogram
            hist_gpu, t_gpu = module.hist1d_gpu(a)
            hist_gpu_t.append(t_gpu)
            # Compute Serial Histogram
            hist_cpu, t_cpu = module.hist1d_cpu(a)
            hist_cpu_t.append(t_cpu)
       
        cl_times_hist.append(np.average(hist_gpu_t))
        py_times_hist.append(np.average(hist_cpu_t))

        print("CPU & GPU computation HIST:", np.allclose(hist_gpu, hist_cpu))

    plt.plot(matrix_size, py_times_hist,'r', label="Python")
    plt.plot(matrix_size, cl_times_hist,'b', label="CL HIST")
    plt.legend(loc='upper left')
    plt.title('1D Histogram')
    plt.xlabel('size of array')
    plt.ylabel('output coding times(sec)')
    # plt.gca().set_xlim((min(matrix_size), max(matrix_size)))
    plt.savefig('CL_HIST_ONLY.png')