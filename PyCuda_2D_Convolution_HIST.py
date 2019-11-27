# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

from __future__ import division
import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray, compiler, tools, cumath
import math

from scipy import signal
from numpy import linalg as la

############### DISCLAIMER #####################
#### When plotting please do one at a time, otherwise the plots for convolution ####
#### and histogram overlap 

class Convolution:
  def __init__(self):
  
    # Write your naive 2D Convolution kernel here
    self.kernel_code_naive = """
        __global__ void convolution_2D_naive(float * in, float * mask, float * out, 
         const int mask_width, int w, int h) {
            int Col = blockIdx.x * blockDim.x + threadIdx.x;
            int Row = blockIdx.y * blockDim.y + threadIdx.y;

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
    # Write your tiled 2D Convolution kernel here
    self.kernel_code_tiled = """
        __global__ void convolution_2D_tiled(float * in, float * mask, float * out, const int mask_width, int w, int h) {
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            int row_o = blockIdx.y*%(TILE_SIZE)s + ty;
            int col_o = blockIdx.x*%(TILE_SIZE)s + tx;

            int row_i = row_o - mask_width/2;
            int col_i = col_o - mask_width/2;

            __shared__ float in_s[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

            if ((row_i >= 0) && (row_i < h) && (col_i >= 0) && (col_i < w)) {
                in_s[ty][tx] = in[row_i * w + col_i];    
            } else {
                in_s[ty][tx] = 0.0f;
            }
    
            __syncthreads(); 
    
            
            if (ty < %(TILE_SIZE)s && tx < %(TILE_SIZE)s) {
                float out_VAL = 0.0f;
                for (int i = 0; i < mask_width; i++) {
                    for (int j = 0; j < mask_width; j++) {
                        out_VAL += mask[i * mask_width + j] * in_s[i + ty][j + tx];
                    }
                }

                __syncthreads();  

                if (row_o < h && col_o < w) {
                    out[row_o * w + col_o] = out_VAL;
                }
            }
        }

    """
    self.kernel_code = self.kernel_code_tiled % {
            'TILE_SIZE': 28,
            'BLOCK_SIZE': 32,
            }
    self.mod_naive = compiler.SourceModule(self.kernel_code_naive)
    self.mod_tiled = compiler.SourceModule(self.kernel_code)
  
  def conv2d_gpu(self, A, kernel):
    # INPUTS:
      # A --> matrix    
      # kernel --> the filter
    
    # Transfer data to device
    self.A_d = gpuarray.to_gpu(A)
    self.kernel_d = gpuarray.to_gpu(kernel)
    self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    
    
    # Compute block and grid size
    grid_dim_x = np.ceil(np.float32(self.A_d.shape[0]/32))
    grid_dim_y = np.ceil(np.float32(self.A_d.shape[1]/32))

    # create CUDA Event to measure time
    start = cuda.Event()
    end = cuda.Event()
    
    # Call kernel function
    func_naive = self.mod_naive.get_function('convolution_2D_naive')

    # Measure time
    start.record()
    start_=time.time()
    func_naive(self.A_d, self.kernel_d, self.output_d, np.int32(kernel.shape[1]), np.int32(A.shape[1]), np.int32(A.shape[0]), 
    block=(32, 32, 1), grid = (np.int(grid_dim_y),np.int(grid_dim_x),1))
    end_ = time.time()
    end.record()

    # CUDA Event synchronize
    end.synchronize()

    output_gpu = self.output_d.get()
    kernel_time = end_-start_

    # Return the result and the kernel execution time
    return output_gpu, kernel_time

  def tiled_conv2d_gpu(self, A, kernel):
    # You can ignore this function for OpenCL
    # INPUTS:
      # A --> matrix
      # kernel --> the filter

    # Transfer data to device
    self.A_d = gpuarray.to_gpu(A)
    self.kernel_d = gpuarray.to_gpu(kernel)
    self.output_d = gpuarray.empty((A.shape[0], A.shape[1]), np.float32)    

    # Compute block and grid size
    grid_dim_x = np.ceil(np.float32(self.A_d.shape[0]/16))
    grid_dim_y = np.ceil(np.float32(self.A_d.shape[1]/16))

    # create CUDA Event to measure time
    start = cuda.Event()
    end = cuda.Event()
    
    # Call kernel function
    func_tiled = self.mod_tiled.get_function('convolution_2D_tiled')

    # Measure time
    start.record()
    start_=time.time()
    func_tiled(self.A_d, self.kernel_d, self.output_d, np.int32(kernel.shape[1]), np.int32(A.shape[1]), np.int32(A.shape[0]), 
    block=(32, 32, 1), grid = (np.int(grid_dim_y),np.int(grid_dim_x),1))
    end_ = time.time()
    end.record()

    output_gpu = self.output_d.get()
    kernel_time = end_-start_

    # Return the result and the kernel execution time
    return output_gpu, kernel_time

  def conv2d_cpu(self, A, kernel):
    # INPUTS:
      # A --> matrix
      # kernel --> the filter

    start = time.time()
    o_cpu = signal.convolve2d(A, kernel, mode='same')
    serial_time = time.time() - start
    output_cpu = o_cpu

    # Return the result and the serial run time
    return output_cpu, serial_time

class Histogram:
  def __init__(self):
    # Write your 1D histogram kernel here
    self.kernel_hist = """ 
    __global__ void hist(const int *buffer, int *hist, const int size)

    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;     

     if (i < size && i >= 0) 
        {
        int bin = buffer[i];
        atomicAdd(&hist[bin], 1);
        }
    }

    """

    self.mod_hist = compiler.SourceModule(self.kernel_hist)

  def hist1d_gpu(self, a, bins=26):
    # INPUTS:
      # a --> 1D array 
      # bins --> the number of bins we want
      #          divide our histogram into
    self.a = a
    self.bins = bins
    a_new = np.floor(self.a.astype(np.float32)/10).astype(np.int32)

    # Transfer data to device
    self.a_gpu = gpuarray.to_gpu(a_new)
    self.out_gpu = gpuarray.zeros((bins,), a_new.dtype)
    
    # Compute block and grid size
    M = len(a)
    
    # create CUDA Event to measure time
    start = cuda.Event()
    end = cuda.Event()

    # Call kernel function
    func_hist = self.mod_hist.get_function('hist')
    
    # Measure time
    start.record()
    start_=time.time()
    func_hist(self.a_gpu, self.out_gpu, np.int32(M), 
    block = (32,1,1), grid = (int(np.ceil(M/32.0)),1,1))
    end_ = time.time()
    end.record()

    hist_gpu = self.out_gpu.get()
    kernel_time = end_-start_

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

    cu_times_naive = []
    py_times = []
    cu_times_tiled = []
    matrix_size = []

  # ITERATIVELY:
    for itr in range(1,40):

    # Create input matrix
        A = np.float32(np.random.randint(low=0, high=9, size=(itr*20,itr*25)))
        matrix_size.append(A.shape[0] * A.shape[1])

    # Create kernel filter
        kernel = np.float32(np.random.randint(low=0, high=5, size=(5,5)))
        kernel_flip = np.rot90(kernel, 2).astype(np.float32)
        
    # Create instance for CUDA
        module = Convolution()    

        naive_times=[]
        serial_times=[]
        tiled_times = []

    # Record times
        for e in range(3):
            # Compute Naive GPU convolution
            cu_output_naive, t = module.conv2d_gpu(A, kernel_flip)
            naive_times.append(t)
            # Compute Tiled GPU Convolution
            cu_output_tiled, t_tiled = module.tiled_conv2d_gpu(A, kernel_flip)
            tiled_times.append(t_tiled)
            # Compute Serial Convolution
            cpu_output, t_cpu = module.conv2d_cpu(A, kernel)
            serial_times.append(t_cpu)
       
        cu_times_naive.append(np.average(naive_times))
        py_times.append(np.average(serial_times))
        cu_times_tiled.append(np.average(tiled_times))
        # print("CUDA OUTPUT NAIVE:", cu_output_naive)
        # print("SERIAL OUTPUT:", cpu_output)

        # print("CUDA OUTPUT SHAPE:", cu_output_naive.shape)
        # print("CUDA Times NAIVE:", cu_times_naive)
        # print()
        
    # Code Equality
        print("CPU & NAIVE GPU computation:", np.allclose(cu_output_naive, cpu_output))
        print("CPU & TILED GPU computation:", np.allclose(cu_output_tiled, cpu_output))
  
  # Plot times
    # MAKE_PLOT = True
    # if MAKE_PLOT:
        # plt.gcf()
    # plt.plot(matrix_size, py_times,'r', label="Python")
    # plt.plot(matrix_size, cu_times_tiled,'b', label="CUDA TILED")
    # plt.plot(matrix_size, cu_times_naive,'g', label="CUDA NAIVE")
    # plt.legend(loc='upper left')
    # plt.title('2D Convolution')
    # plt.xlabel('size of array')
    # plt.ylabel('output coding times(sec)')
    # plt.gca().set_xlim((min(matrix_size), max(matrix_size)))
    # plt.savefig('plots_pycuda_2D_conv_SMALL.png')
  ##########################################################
  #                    HISTOGRAM                           #
  ##########################################################
    cu_times_hist = []
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
       
        cu_times_hist.append(np.average(hist_gpu_t))
        py_times_hist.append(np.average(hist_cpu_t))

        print("CPU & GPU computation HIST:", np.allclose(hist_gpu, hist_cpu))

  # Plot times
    # MAKE_PLOT = True
    # if MAKE_PLOT:
        # plt.gcf()
    plt.plot(matrix_size, py_times_hist,'r', label="Python")
    plt.plot(matrix_size, cu_times_hist,'b', label="CUDA HIST")
    plt.legend(loc='upper left')
    plt.title('1D Histogram')
    plt.xlabel('size of array')
    plt.ylabel('output coding times(sec)')
    # plt.gca().set_xlim((min(matrix_size), max(matrix_size)))
    plt.savefig('HIST_ONLY.png')