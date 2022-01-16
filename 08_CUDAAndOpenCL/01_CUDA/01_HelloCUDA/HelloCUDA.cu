//Headers

#include<stdio.h>
#include<cuda.h>

//Global Variables
int inputLength = 5;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

//global kernel function definition
__global__ void vecAdd(float *in1, float *in2, float *out, int len)
{
	//variable declaration
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//code
	if(i<len)
	{	
		out[i] = in1[i] + in2[i];
	}
}

int main(int argc, char *argv[])
{
	//function declaration
	void cleanup(void);

	//code
	//allocate host-memory
	hostInput1 = (float *)malloc(inputLength * sizeof(float));
	if(hostInput1 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 1.\nExitting....\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float *)malloc(inputLength * sizeof(float));
	if(hostInput2 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 2.\nExitting....\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float *)malloc(inputLength * sizeof(float));
	if(hostOutput == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Array.\nExitting....\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//fill above input host vectors with arbitary but hard coded data
	hostInput1[0] = 101.0;
	hostInput1[1] = 102.0;
	hostInput1[2] = 103.0;
	hostInput1[3] = 104.0;
	hostInput1[4] = 105.0;

	hostInput2[0] = 201.0;
	hostInput2[1] = 202.0;
	hostInput2[2] = 203.0;
	hostInput2[3] = 204.0;
	hostInput2[4] = 205.0;

	//allocate device(GPU) memory
	int s = inputLength * sizeof(float);
	cudaError_t err = cudaSuccess;

	err = cudaMalloc((void **)&deviceInput1, s);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&deviceInput2, s);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&deviceOutput, s);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//copy host memory contents to device memory contents
	err = cudaMemcpy(deviceInput1, hostInput1, s, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(deviceInput2, hostInput2, s, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//cuda kernel configuration
	dim3 DimGrid = dim3(ceil(inputLength / 256.0), 1, 1);
	dim3 DimBlock = dim3(256, 1, 1);
	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

	//copy device memory to host memory
	err = cudaMemcpy(hostOutput, deviceOutput, s, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting....\n", cudaGetErrorString(err), __FILE__, __LINE__);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//result
	int i;
	for(i = 0; i < inputLength; i++)
	{
		printf("%f + %f = %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
	}

	//total cleanup
	cleanup();

	return(0);
}

void cleanup(void)
{
	//code

	//free allocated memory
	if(deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceInput1 = NULL;
	}
	
	if(deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceInput2 = NULL;
	}

	if(deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}

	//free GPU Memory

	if(hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}
	
	if(hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}

	if(hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}
}
