#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess) std::cout << cudaGetErrorString(error) << std::endl;}

__global__ void kernelSumCurrents(cudaPitchedPtr fieldJ, float3 *gCurrent, dim3 superCellSize)
{
    __shared__ float3 sh_sumJ;
    __syncthreads();

    const int linearThreadIdx = threadIdx.x + 8*threadIdx.y + 8*8*threadIdx.z;
    if(linearThreadIdx == 0)
    {
      sh_sumJ.x = 0.0; sh_sumJ.y = 0.0; sh_sumJ.z = 0.0;
    }
    __syncthreads();
    dim3 superCellIdx;
    superCellIdx.x = blockIdx.x/16 + 1;
    superCellIdx.y = blockIdx.y + 1;
    superCellIdx.z = blockIdx.x%16 + 1;
    
    dim3 cell;
    cell.x = superCellIdx.x * superCellSize.x + threadIdx.x;
    cell.y = superCellIdx.y * superCellSize.x + threadIdx.y;
    cell.z = superCellIdx.z * superCellSize.x + threadIdx.z;

    char *fieldJPtr = (char *)fieldJ.ptr;
    size_t jSlicePitch = fieldJ.pitch * fieldJ.ysize;
    char *jSlice = fieldJPtr + cell.z * jSlicePitch;
    float3 *jRow = (float3 *)(jSlice + cell.y * fieldJ.pitch);
    const float3 myJ = jRow[cell.x];

    atomicAdd(&(sh_sumJ.x), myJ.x); 
    atomicAdd(&(sh_sumJ.y), myJ.y); 
    atomicAdd(&(sh_sumJ.z), myJ.z); 
    
#if 0
    __syncthreads();

    if (linearThreadIdx == 0)
    {
	    //atomicAdd(&(gCurrent->x), sh_sumJ.x); 
	    //atomicAdd(&(gCurrent->y), sh_sumJ.y); 
	    //atomicAdd(&(gCurrent->z), sh_sumJ.z); 
	    atomicAdd(&(gCurrent->x), myJ.x); 
	    atomicAdd(&(gCurrent->y), myJ.y); 
	    atomicAdd(&(gCurrent->z), myJ.z); 
    }
#endif
}

int main(){
    cudaExtent extent;
    cudaPitchedPtr d_field_j;
    extent.width = 144 * sizeof(float3); //80x80x72, 144x80x72, 144x144x72, 144x144x136
    extent.height = 144;
    extent.depth = 72;
    CUDA_CHECK(cudaMalloc3D(&d_field_j, extent));
    CUDA_CHECK(cudaMemset3D(d_field_j, 0, extent));
    float3 *d_sum_currents;
    size_t sum_currents_pitch = 1;
    CUDA_CHECK(cudaMallocPitch(&d_sum_currents, &sum_currents_pitch, sizeof(float3), 1));
    dim3 grid(256, 16, 1), block(8, 8, 4), super_cell_size(8, 8, 4);
    struct timeval prev, now;
    for(int i=0; i<10; i++){
	     gettimeofday(&prev, NULL);
	    kernelSumCurrents<<<grid, block>>>(d_field_j, d_sum_currents, super_cell_size);
	     CUDA_CHECK(cudaDeviceSynchronize());
	     gettimeofday(&now, NULL);
	     std::cout << "Analysis kernel running time => " << (now.tv_sec-prev.tv_sec)*1000000 + (now.tv_usec-prev.tv_usec) << std::endl;
    }
    return 0;
}

