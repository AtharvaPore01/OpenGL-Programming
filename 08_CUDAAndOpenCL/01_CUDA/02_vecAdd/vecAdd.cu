//headers

#include <stdio.h>
#include <cuda.h>
#include "helper_timer.h"

//global variables
int iNumberOfArrayElements = 11444777;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

float timeOnCPU;
float timeOnGPU;

//cuda kernel

__global__ void vecAdd(float *in1, float *in2, float *out, int len)
{
	//variable declaration
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//code
	if(i < len)
	{
		out[i] = in1[i] + in2[i];
	}

}

int main(int argc, char *argv[])
{
	//function declaration
	void FillFloatArrayWithRandomNumbers(float *, int);
	void vecAddHost(const float*, const float *, float *, int);
	void CleanUp(void);

	//code
	//allocate host memory
	hostInput1 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
	if(hostInput1 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate The Memory For Host Input Array 1.\nExitting....\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
	if(hostInput2 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate The Memory For Host Input Array 2.\nExitting....\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float *)malloc(sizeof(float) * iNumberOfArrayElements);
	if(hostOutput == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate The Memory For Host Output Array.\nExitting....\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(sizeof(float) * iNumberOfArrayElements);
	if(gold == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate The Memory For Gold.\nExitting....\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//fill above vectors with arbitary but hard coded data 
	FillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	FillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	//allocate device(GPU) memory
	cudaError_t err = cudaSuccess;
	int size = sizeof(float) * iNumberOfArrayElements;

	err = cudaMalloc((void **)&deviceInput1, size);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&deviceInput2, size);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&deviceOutput, size);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//copy host memory content in device memory(GPU memory)
	err = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//cuda kernel configuration
	dim3 DimGrid = dim3(ceil(iNumberOfArrayElements / 256.0), 1, 1);
	dim3 DimBlock = dim3(256, 1, 1);

	//start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

	//stop timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	//copy Device Memory to CPU memory
	err = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//results
	vecAddHost(hostInput1, hostInput2, gold, iNumberOfArrayElements);

	//compare results for golden-host
	const float epsilon = 0.000001f;
	bool bAccuracy = true;
	int breakValue = 0;
	int i;
	for(i = 0; i < iNumberOfArrayElements; i++)
	{
		float val1 = gold[i];
		float val2 = hostOutput[i];
		if(fabs(val1 - val2) > epsilon)
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	if(bAccuracy == false)
	{
		printf("Break value : %d\n", breakValue);
	}

	char str[125];
	if(bAccuracy == true) 
	{
		sprintf(str, "%s", "Comparison OF Output Arrays On CPU & GPU Are Accurate Within The Limit Of 0.000001");
	}
	else
	{
		sprintf(str, "%s", "Not All Comparison OF Output Arrays On CPU & GPU Are Accurate Within The Limit Of 0.000001");	
	}

	printf("[1]\t1st Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
	printf("[2]\t2nd Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
	printf("[3]\tGrid Dimensions : (%d, 1, 1)\n\n", DimGrid.x);
	printf("[4]\tBlock Dimensions : (%d, 1, 1)\n\n", DimBlock.x);
	printf("[5]\tSum Of Each OF Array Elements Above 2 Arrays Creates 3rd Array As : \n\n");
	printf("   \t3rd Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
	printf("[6]\tTime Taken By CPU And GPU : \n\n");
	printf("   \tTime Taken For Above Addtion On CPU : %.6f (ms)\n", timeOnCPU);
	printf("   \tTime Taken For Above Addtion On GPU : %.6f (ms)\n", timeOnGPU);
	printf("\n");
	printf("%s\n\n", str);

	//total cleanup
	CleanUp();
	return(0);
} 

void FillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
	//code
	int i;
	const float fScale = 1.0f / (float)RAND_MAX;
	for(i = 0;i < iSize; i++)
	{
		pFloatArray[i] = fScale * rand();
	}

}

void vecAddHost(const float *pFloatData1, const float *pFloatData2, float *pFloatResult, int iNumElements)
{
	int i;

	//Start Timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for(i = 0; i < iNumElements; i++)
	{
		pFloatResult[i] = pFloatData1[i] + pFloatData2[i];
	}

	//stop timer
	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

}
void CleanUp(void)
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

	if(gold)
	{
		free(gold);
		gold = NULL;
	}

}
