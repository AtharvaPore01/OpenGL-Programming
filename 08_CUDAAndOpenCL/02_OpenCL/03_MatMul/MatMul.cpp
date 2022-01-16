#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<CL/opencl.h>

#include"helper_timer.h"

//global opencl variables

cl_int ret_ocl;
cl_platform_id oclPlatformID;
cl_device_id oclComputeDeviceID;
cl_context oclContext;
cl_command_queue oclCommandQueue;
cl_program oclProgram;
cl_kernel oclKernel;

char *oclSourceCode = NULL;
size_t sizeKernelCodeLength;

size_t localWorkSize = 256;
size_t globalWorkSize;

float *hostA = NULL;
float *hostB = NULL;
float *hostC = NULL;
float *CHost = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU;
float timeOnGPU;

int main(void)
{
	//function declaration
	void FillFloatArrayWithRandomNumbers(float *, int);
	void CleanUp(void);
	size_t Round_Global_Size_To_Nearest_Multiple_Of_Local_Size(int, unsigned int);
	void matMulHost(float *, float *, float *, int, int, int);
	char* loadOclProgramSource(const char *, const char *, size_t *);
	
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

	//for device
	numCRows = numARows;
	numCColumns = numBColumns;
	//for host
	numCHostRows = numARows;
	numCHostColumns = numBColumns;

	int sizeA = numARows * numAColumns * sizeof(float);
	int sizeB = numBRows * numBColumns * sizeof(float);
	int sizeC = numCRows * numCColumns * sizeof(float);
	int sizeCHost = numCHostRows * numCHostColumns * sizeof(float);

	//allocate host memory
	hostA = (float *)malloc(sizeA);
	if (hostA == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix A.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	hostB = (float *)malloc(sizeB);
	if (hostB == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix B.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	hostC = (float *)malloc(sizeC);
	if (hostC == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix C.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	CHost = (float *)malloc(sizeCHost);
	if (CHost == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix CHost.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}
	//fill above input host with arbitary but hard coded values
	FillFloatArrayWithRandomNumbers(hostA, numARows * numAColumns);
	FillFloatArrayWithRandomNumbers(hostB, numBRows * numBColumns);

	//get opencl supported platform ids
	ret_ocl = clGetPlatformIDs(1, &oclPlatformID, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetPlatformIDs() Failed : %d.\n Exitting...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//get opencl supported GPU device ids
	ret_ocl = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclComputeDeviceID, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetDeviceIDs() Failed : %d.\n Exitting...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	char gpu_name[255];
	clGetDeviceInfo(oclComputeDeviceID, CL_DEVICE_NAME, sizeof(gpu_name), &gpu_name, NULL);
	printf("*****************************************%s***************************************************\n\n", gpu_name);

	//create opencl compute context
	oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateContext() Failed : %d.\n Exitting...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create Command queue
	oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateCommandQueue() Failed : %d.\n Exitting...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create opencl program from .cl
	oclSourceCode = loadOclProgramSource("MatMul.cl", "", &sizeKernelCodeLength);
	cl_int status = 0;
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&oclSourceCode, &sizeKernelCodeLength, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateProgramWithSource() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//build opencl program
	ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clBuildProgram() Failed : %d.\n Exitting Now...\n", ret_ocl);

		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("OpenCL Program Build Log : %s\n", buffer);

		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create opencl kernel by passing kernel funcion name that we used in .cl
	oclKernel = clCreateKernel(oclProgram, "matrixMultiplication", &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateKernel() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	int Size = (numCRows * numCColumns) * sizeof(cl_float);
	deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//set opencl kernel arguments. Out OpenCL has 4 arguments 0, 1, 2, 3
	//set 0 based 0th argument i.e. deviceInput1
	ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceA);
	//deviceInput1 will get mapped to in1 in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceB);
	//deviceInput2 will get mapped to in2 in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceC);
	//deviceOutput will get mapped to out in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&numARows);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void *)&numAColumns);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void *)&numBRows);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void *)&numBColumns);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 7, sizeof(cl_int), (void *)&numCRows);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 8, sizeof(cl_int), (void *)&numCColumns);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//write above input device buffer to device memory
	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, Size, hostA, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueWriteBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, Size, hostB, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueWriteBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	globalWorkSize = Round_Global_Size_To_Nearest_Multiple_Of_Local_Size(localWorkSize, (numCRows * numCColumns));

	//start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueNDRangeKernel() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	
	//finish ocl Command queue 
	clFinish(oclCommandQueue);

	//Stop Timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	//transfer result from device to host
	ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, Size, hostC, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
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
	printf("[3]\tSum Of Each OF Array Elements Above 2 Matrices Creates 3rd Matrix As : \n\n");
	printf("   \t3rd Matrix Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostC[0], (numCRows * numCColumns) - 1, hostC[(numCRows * numCColumns) - 1]);
	printf("[4]\tTime Taken By CPU And GPU : \n\n");
	printf("   \tTime Taken For Above Addtion On CPU : %.6f (ms)\n", timeOnCPU);
	printf("   \tTime Taken For Above Addtion On GPU : %.6f (ms)\n", timeOnGPU);
	printf("\n");
	printf("%s\n\n", str);

	//total cleanup
	CleanUp();

	return(0);
}

char *loadOclProgramSource(const char *fileName, const char *preamble, size_t *sizeFinalLength)
{
	//locals
	FILE *pFile = NULL;
	size_t sizeSourceLength;

	pFile = fopen(fileName, "rb");
	if(pFile == NULL)
	{
		return(NULL);
	}

	size_t sizePreambleLength = (size_t)strlen(preamble);

	//get the length of the source code
	fseek(pFile, 0, SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	char *sourceString = (char *)malloc(sizeSourceLength + sizePreambleLength + 1);
	memcpy(sourceString, preamble, sizePreambleLength);

	if(fread((sourceString) + sizePreambleLength, sizeSourceLength, 1, pFile) != 1)
	{
		fclose(pFile);
		free(sourceString);
		return(0);
	}

	//clode the file and return
	fclose(pFile);
	if(sizeFinalLength != 0)
	{
		*sizeFinalLength = sizeSourceLength + sizePreambleLength;
	}

	sourceString[sizeSourceLength + sizePreambleLength] = '\0';
	return(sourceString);

}

void FillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
	//code
	int i;
	const float fScale = 1.0f / (float)RAND_MAX;
	for(i = 0; i < iSize; i++)
	{
		pFloatArray[i] = fScale * rand();
	}
}

size_t Round_Global_Size_To_Nearest_Multiple_Of_Local_Size(int local_size, unsigned int global_size)
{
	//code
	unsigned int r = global_size % local_size;
	if(r == 0)
	{
		return(global_size);
	}
	else
	{
		return(global_size + local_size - r);
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

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

void CleanUp(void)
{
	//code
	if(deviceC)
	{
		clReleaseMemObject(deviceC);
		deviceC = NULL;
	}

	if(deviceB)
	{
		clReleaseMemObject(deviceB);
		deviceB = NULL;
	}

	if(deviceA)
	{
		clReleaseMemObject(deviceA);
		deviceA = NULL;
	}

	if(oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}

	if(oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if(oclSourceCode)
	{
		free((void *)(oclSourceCode));
		oclSourceCode = NULL;
	}

	if(oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if(oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
	}

	if(hostC)
	{
		free(hostC);
		hostC = NULL;
	}

	if(hostB)
	{
		free(hostB);
		hostB = NULL;
	}

	if(hostA)
	{
		free(hostA);
		hostA = NULL;
	}

	if(CHost)
	{
		free(CHost);
		CHost = NULL;
	}
}

