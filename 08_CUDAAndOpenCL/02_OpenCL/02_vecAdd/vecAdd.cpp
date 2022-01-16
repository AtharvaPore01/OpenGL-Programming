//headers

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<CL/opencl.h>

#include"helper_timer.h"

//global OpenCL Variables

cl_int ret_ocl;
cl_platform_id oclPlatformID;
cl_device_id oclComputeDeviceID;	//compute device id
cl_context oclContext;				//compute context
cl_command_queue oclCommandQueue;	//compute command queue
cl_program oclProgram;				//compute program
cl_kernel oclKernel;				//compute kernel

char *oclSourceCode = NULL;
size_t sizeKernelCodeLength;

//11444777

int iNumberOfArrayElements = 11444777;
size_t localWorkSize = 256;
size_t globalWorkSize;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

float timeOnCPU;
float timeOnGPU;

int main(void)
{
	//function declaration
	void FillFloatArrayWithRandomNumbers(float *, int);
	size_t Round_Global_Size_To_Nearest_Multiple_Of_Local_Size(int, unsigned int);
	void vecAddHost(const float *, const float *, float *, int);
	void CleanUp(void);
	char *loadOCLProgramSource(const char *, const char *, size_t *);

	//code
	int size = iNumberOfArrayElements * sizeof(float);

	//allocate host memory
	hostInput1 = (float *)malloc(size);
	if (hostInput1 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 1.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}
	hostInput2 = (float *)malloc(size);
	if (hostInput2 == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 2.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}
	hostOutput = (float *)malloc(size);
	if (hostOutput == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Array.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(size);
	if(gold == NULL)
	{
		printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Gold Output Array.\n Exitting...\n");
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//fill above host hoist vectors with arbitary but hard coded data
	FillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	FillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	//get opencl supporting platform id
	ret_ocl = clGetPlatformIDs(1, &oclPlatformID, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetPlatformIDs() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//get opencl supported gpu device's id
	ret_ocl = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclComputeDeviceID, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetDeviceIDs() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create opencl compute context
	oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateContext() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create command queue
	oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateCommandQueue() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//create opencl program for .cl
	oclSourceCode = loadOCLProgramSource("VecAdd.cl", "", &sizeKernelCodeLength);

	const char *szOpenCLKernelPath = "VecAdd.cl";
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&oclSourceCode, NULL, &ret_ocl);
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

	//create opencl kernel by passing kernel function name that we used in .cl
	oclKernel = clCreateKernel(oclProgram, "vecAdd", &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateKernel() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	int Size = iNumberOfArrayElements * sizeof(cl_float);
	//allocat device memory
	deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, Size, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clCreateBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//set opencl kernel arguments. Out OpenCL has 4 arguments 0, 1, 2, 3
	//set 0 based 0th argument i.e. deviceInput1
	ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
	//deviceInput1 will get mapped to in1 in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
	//deviceInput2 will get mapped to in2 in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
	//deviceOutput will get mapped to out in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&iNumberOfArrayElements);
	//InputLength will get mapped to len in .cl file
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clSetKernelArg() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	
	//write above input device buffer to device memory
	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceInput1, CL_FALSE, 0, Size, hostInput1, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueWriteBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, Size, hostInput2, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueWriteBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//run the kernel

	globalWorkSize = Round_Global_Size_To_Nearest_Multiple_Of_Local_Size(localWorkSize, iNumberOfArrayElements);

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
	ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, Size, hostOutput, 0, NULL, NULL);
	if(ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d.\n Exitting Now...\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	//result

	vecAddHost(hostInput1, hostInput2, gold, iNumberOfArrayElements);

	//compare the result
	const float epsilon = 0.000001;
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
	printf("[1]\t1st Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
	printf("[2]\t2nd Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
	printf("[3]\tSum Of Each OF Array Elements Above 2 Arrays Creates 3rd Array As : \n\n");
	printf("   \t3rd Array Is From 0th Element %.6f To %dth Element %.6f.\n\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
	printf("[4]\tTime Taken By CPU And GPU : \n\n");
	printf("   \tTime Taken For Above Addtion On CPU : %.6f (ms)\n", timeOnCPU);
	printf("   \tTime Taken For Above Addtion On GPU : %.6f (ms)\n", timeOnGPU);
	printf("\n");
	printf("%s\n\n", str);

	//total cleanup
	CleanUp();
	return(0);
}

char *loadOCLProgramSource(const char *fileName, const char *preamble, size_t *sizeFinalLength)
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

void vecAddHost(const float *pFloatData1, const float *pFloatData2, float *pFloatResult, int iNumElements)
{
	int i;

	//start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for(i = 0; i < iNumElements; i++)
	{
		pFloatResult[i] = pFloatData1[i] + pFloatData2[i];
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

void CleanUp(void)
{
	//code
	if(deviceOutput)
	{
		clReleaseMemObject(deviceOutput);
		deviceOutput = NULL;
	}

	if(deviceInput2)
	{
		clReleaseMemObject(deviceInput2);
		deviceInput2 = NULL;
	}

	if(deviceInput1)
	{
		clReleaseMemObject(deviceInput1);
		deviceInput1 = NULL;
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

	if(hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}

	if(hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}

	if(hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}

	if(gold)
	{
		free(gold);
		gold = NULL;
	}
}
