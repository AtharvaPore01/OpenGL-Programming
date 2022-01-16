//headers

#include <stdio.h>
#include <cuda.h>
#include "helper_timer.h"

#define BLOCK_WIDTH 4

//variable declaration

float *hostA = NULL;
float *hostB = NULL;
float *hostC = NULL;
float *CHost = NULL;

float *deviceA = NULL;
float *deviceB = NULL;
float *deviceC = NULL;

float timeOnCPU;
float timeOnGPU;

//global kernel function

__global__ void matrixMultiplication(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//variable declaration

	int row = blockIdx.x * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//code
	if((row < numARows) && (col < numBColumns))
	{
		float Cvalue = 0.0;
		for(int k = 0; k < numAColumns; k++)
		{
			Cvalue += A[row * numAColumns + k] * B[k * numBColumns + col];
		}
		C[row * numCColumns + col] = Cvalue;
	}
}

int main(int argc, char *argv[])
{
	//function declaration
	void FillFloatArrayWithRandomNumbers(float *, int);
	void matMulHost(float *, float *, float *, int, int, int);
	void CleanUp(void);

	//variable declaration

	int numARows;
	int numAColumns;
	int numBRows;
	int numBColumns;
	int numCRows;
	int numCColumns;
	int numCHostRows;
	int numCHostColumns;

	//code
	numARows = 4;
	numAColumns = 4;
	numBRows = 4;
	numBColumns = 4;

	numCRows = numARows;
	numCColumns = numBColumns;

	numCHostRows = numARows;
	numCHostColumns = numBColumns;

	int sizeA = numARows * numAColumns * sizeof(float);
	int sizeB = numBRows * numBColumns * sizeof(float);
	int sizeC = numCRows * numCColumns * sizeof(float);
	int sizeCHost = numCHostRows * numCHostColumns * sizeof(float);

	//allocate host memory
	hostA = (float *)malloc(sizeA);
	if(hostA == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix A.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}
	
	hostB = (float *)malloc(sizeB);
	if(hostB == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix B.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}
	
	hostC = (float *)malloc(sizeC);
	if(hostC == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix C.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	CHost = (float *)malloc(sizeCHost);
	if(CHost == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix CHost.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//fill above input host with arbitary but hard coded values
	FillFloatArrayWithRandomNumbers(hostA, numARows * numAColumns);
	FillFloatArrayWithRandomNumbers(hostB, numBRows * numBColumns);

	//allocate device memory
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&deviceA, sizeA);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&deviceB, sizeB);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&deviceC, sizeC);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//copy host memory contents to device memory
	err = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//cuda kernel configuration
	dim3 DimGrid = dim3(ceil((int)numCColumns / (int)BLOCK_WIDTH), ceil((int)numCRows / (int)BLOCK_WIDTH), 1);
	dim3 DimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	//start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	matrixMultiplication<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

	//stop timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	//copy device memory to host
	err = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\n Exitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//result

	matMulHost(hostA, hostB, CHost, numAColumns, numCHostRows, numCHostColumns);

	//compare result for golden-host
	const float epsilon = 0.000001f;
	bool bAccuracy = true;
	int breakValue = 0;
	int i;

	for(i = 0; i < numARows * numAColumns; i++)
	{
		float val1 = CHost[i];
		float val2 = hostC[i];
		if(fabs(val1 - val2) > epsilon)
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	if(bAccuracy == false)
	{
		printf("Break Value = %d\n", breakValue);
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

	printf("[1]\t1st Matrix Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostA[0], (numARows * numAColumns) - 1, hostA[(numARows * numAColumns) - 1]);
	printf("[2]\t2nd Matrix Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostB[0], (numBRows * numBColumns) - 1, hostB[(numBRows * numBColumns) - 1]);
	printf("[3]\tGrid Dimensions : (%d, 1, 1)\n\n", DimGrid.x);
	printf("[4]\tBlock Dimensions : (%d, 1, 1)\n\n", DimBlock.x);
	printf("[5]\tSum Of Each OF Array Elements Above 2 Matrices Creates 3rd Matrix As : \n\n");
	printf("   \t3rd Matrix Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostC[0], (numCRows * numCColumns) - 1, hostC[(numCRows * numCColumns) - 1]);
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

void matMulHost(float *A, float *B, float *C, int iAColumns, int iCRows, int iCColumns)
{
	//code
	//start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for(int i = 0; i < iCRows; ++i)
	{
		for(int j = 0; j < iCColumns; ++j)
		{
			float sum = 0.0f;
			for(int k = 0; k < iAColumns; ++k)
			{
				float a = A[i * iAColumns + k];
				float b = B[k * iCColumns + j];
				sum += a * b;
			}
			C[i * iCColumns + j] = sum;
		}
	}

	//stop timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

}

void CleanUp(void)
{
		//code

	//free allocated memory
	if(deviceA)
	{
		cudaFree(deviceA);
		deviceA = NULL;
	}
	
	if(deviceB)
	{
		cudaFree(deviceB);
		deviceB = NULL;
	}

	if(deviceC)
	{
		cudaFree(deviceC);
		deviceB = NULL;
	}

	//free GPU Memory

	if(hostA)
	{
		free(hostA);
		hostA = NULL;
	}
	
	if(hostB)
	{
		free(hostB);
		hostB = NULL;
	}

	if(hostC)
	{
		free(hostC);
		hostC = NULL;
	}

	if(CHost)
	{
		free(CHost);
		CHost = NULL;
	}

}
