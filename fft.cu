#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <sys/time.h>

#define INPUT_SIZE 5120

int main(){

	struct timeval start, end;
	cudaError_t err;
	cufftResult res;

	double *idata = (double *)malloc(INPUT_SIZE * sizeof(double));
	for(int i=0; i<INPUT_SIZE; i++){
		idata[i] = rand() / (double)RAND_MAX;
	}
	double *odata = (double *)malloc(2*INPUT_SIZE * sizeof(double));
	for(int i=0; i<2*INPUT_SIZE; i++){
		odata[i] = 0.0;
	}

	double *d_idata, *d_odata;
	err = cudaMalloc((void **)&d_idata, INPUT_SIZE*sizeof(double));
	if(err != cudaSuccess){
		printf("cudaMalloc failed\n");
	}
	err = cudaMalloc((void **)&d_odata, 2*INPUT_SIZE*sizeof(double));
	if(err != cudaSuccess){
		printf("cudaMalloc failed\n");
	}

	cufftHandle plan;
	res = cufftPlan1d(&plan, INPUT_SIZE, CUFFT_R2C, 1);
	if(res != CUFFT_SUCCESS){
		printf("cufftPlan1d failed\n");
	}

	for(int i=0; i<10; i++){
        gettimeofday(&start, NULL);
	err = cudaMemcpy(d_idata, idata, INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("cudaMemcpyHostToDevice failed\n");
	}
        gettimeofday(&end, NULL);
	printf("cudaMemcpyHostToDevice took %llu us \n", (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

        gettimeofday(&start, NULL);
	res = cufftExecR2C(plan, (cufftReal *)d_idata, (cufftComplex *)d_odata);
	if(res != CUFFT_SUCCESS){
		printf("cufftExecR2C failed\n");
	}
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("cudaDeviceSynchronize failed\n");
	}
        gettimeofday(&end, NULL);
	printf("cufftExecR2C took %llu us \n", (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

        gettimeofday(&start, NULL);
	err = cudaMemcpy(odata, d_odata, 2*INPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("cudaMemcpyDeviceToHost failed\n");
	}
        gettimeofday(&end, NULL);
	printf("cudaMemcpyDeviceToHost took %llu us \n", (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	}

	return 0;

}
