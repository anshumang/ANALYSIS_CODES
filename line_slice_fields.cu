#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess) std::cout << cudaGetErrorString(error) << std::endl;}

__global__ void kernelLineSliceFields(cudaPitchedPtr fieldE, cudaPitchedPtr fieldB, float3 *sliceDataField, dim3 globalCellIdOffset, dim3 globalNrOfCells, dim3 superCellSize, int guardingSuperCells)
{
    dim3 superCellIdx;
    //superCellIdx(mapper.getSuperCellIndex(blockIdx))
    superCellIdx.x = blockIdx.x/16 + 1; 
    superCellIdx.y = blockIdx.y + 1; 
    superCellIdx.z = blockIdx.x%16 + 1; 
    if((threadIdx.x==0)&&(threadIdx.y==0)&&(threadIdx.z==0)){
	//printf("Block %d %d %d => Supercell %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, superCellIdx.x, superCellIdx.y, superCellIdx.z);
    }

    __syncthreads();

    dim3 localCell;
    //dim3 superCellSize;
    //superCellSize.x = 8; superCellSize.y = 8; superCellSize.z = 4;
    localCell.x = superCellIdx.x * superCellSize.x + threadIdx.x;
    localCell.y = superCellIdx.y * superCellSize.y + threadIdx.y;
    localCell.z = superCellIdx.z * superCellSize.z + threadIdx.z;
    if((threadIdx.x==0)&&(threadIdx.y==0)&&(threadIdx.z==0)){
	//printf("Supercell %d %d %d => Localcell %d %d %d\n", superCellIdx.x, superCellIdx.y, superCellIdx.z, localCell.x, localCell.y, localCell.z);
    }
    if((localCell.x==64)&&(localCell.z==64)){
	printf("X==64 && Z==64 : %d %d %d => %d %d %d => %d %d %d\n", superCellIdx.x, superCellIdx.y, superCellIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, localCell.x, localCell.y, localCell.z);
    }

    char *fieldEPtr = (char *)fieldE.ptr;
    size_t eSlicePitch = fieldE.pitch * fieldE.ysize;
    char *eSlice = fieldEPtr + localCell.z * eSlicePitch;
    //float3 *eRow = (float3 *)eSlice + localCell.y * fieldE.pitch;
    float3 *eRow = (float3 *)((char *)eSlice + localCell.y * fieldE.pitch);
    float3 e = eRow[localCell.x];   

    char *fieldBPtr = (char *)fieldB.ptr;
    size_t bSlicePitch = fieldB.pitch * fieldB.ysize;
    char *bSlice = fieldBPtr + localCell.z * bSlicePitch;
    //float3 *bRow = (float3 *)bSlice + localCell.y * fieldB.pitch;
    float3 *bRow = (float3 *)((char *)bSlice + localCell.y * fieldB.pitch);
    float3 b = bRow[localCell.x];   

    dim3 localCellWG;
    //int guardingSuperCells = 1;
    localCellWG.x = localCell.x - superCellSize.x * guardingSuperCells; 
    localCellWG.y = localCell.y - superCellSize.y * guardingSuperCells; 
    localCellWG.z = localCell.z - superCellSize.z * guardingSuperCells; 

    dim3 globalCell;
    globalCell.x = localCellWG.x + globalCellIdOffset.x; 
    globalCell.y = localCellWG.y + globalCellIdOffset.y; 
    globalCell.z = localCellWG.z + globalCellIdOffset.z; 

    if(globalCell.x == globalNrOfCells.x /2){
	if(globalCell.z == globalNrOfCells.z /2){
	     printf("%d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
	     sliceDataField[localCellWG.y] = e;
        }
    }

    __syncthreads();
}

int main(){
     cudaExtent extent;
     cudaPitchedPtr d_field_e, d_field_b;
     extent.width = 960 * sizeof(float3);
     extent.height = 80;
     extent.depth = 72;
     CUDA_CHECK(cudaMalloc3D(&d_field_e, extent));//960 80 72
     CUDA_CHECK(cudaMalloc3D(&d_field_b, extent));//960 80 72
     CUDA_CHECK(cudaMemset3D(d_field_e, 0, extent));
     CUDA_CHECK(cudaMemset3D(d_field_b, 0, extent));
     float3 *d_slice_data_field;
     size_t slice_data_field_pitch = 1;
     CUDA_CHECK(cudaMallocPitch(&d_slice_data_field, &slice_data_field_pitch, 64 * sizeof(float3), 1));
     dim3 grid(128, 8, 1), block(8, 8, 4);
     dim3 global_cell_id_offset(0, 0, 0), global_nr_of_cells(128, 128, 128), super_cell_size(8, 8, 4);
     int guarding_super_cells = 1;
     struct timeval start, end;
     for(int i=0; i<25; i++){ //-s 25 -lslice.period 1
     //wait_for();
     gettimeofday(&start, NULL);
     kernelLineSliceFields<<<grid, block>>>(d_field_e, d_field_b, d_slice_data_field, global_cell_id_offset, global_nr_of_cells, super_cell_size, guarding_super_cells);
     CUDA_CHECK(cudaDeviceSynchronize());
     gettimeofday(&end, NULL);
     std::cout << "kernelLineSliceFields execution time (in us) : " << end.tv_sec + end.tv_usec - start.tv_sec - start.tv_usec << std::endl;
     //release();
     }
     return 0;
}
