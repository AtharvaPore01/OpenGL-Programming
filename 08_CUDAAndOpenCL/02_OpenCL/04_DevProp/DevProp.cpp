//Headers
#include <stdio.h>
#include <stdlib.h>

//opencl headers
#include<CL/opencl.h>

int main(void)
{
	//function declaration
	void PrintOpenCLDeviceProperties(void);

	//code
	PrintOpenCLDeviceProperties();
}

void PrintOpenCLDeviceProperties(void)
{
	//Variable declaration

	cl_int ret_ocl;
	cl_platform_id oclPlatformID;
	cl_uint dev_count;
	cl_device_id *ocl_device_ids;

	char oclPlatformInfo[512];

	//code
	printf("\n\n");
	printf("OpenCL Inforamation : \n\n");
	printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");

	//get first platform id
	ret_ocl = 
	ret_ocl = clGetPlatformIDs(1, &oclPlatformID, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetPlatformIDs() Failed : %d.\n Exitting Now...\n", ret_ocl);
		exit(EXIT_FAILURE);
	}

	ret_ocl = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error - clGetDeviceIDs() Failed : %d.\n Exitting Now...\n", ret_ocl);
		exit(EXIT_FAILURE);
	}
	else if(dev_count == 0)
	{
		printf("There Is No OpenCL Supported Device On This System.\nExitting Now....\n");
		exit(EXIT_FAILURE);
	}
	else
	{
		//get platform id
		clGetPlatformInfo(oclPlatformID, CL_PLATFORM_NAME, 500, &oclPlatformInfo, NULL);
		printf("[1]\tOpenCL Supported GPU Platform Name                                                               : %s\n\n", oclPlatformInfo);

		//get platform version
		clGetPlatformInfo(oclPlatformID, CL_PLATFORM_VERSION, 500, &oclPlatformInfo, NULL);
		printf("[2]\tOpenCL Supported GPU Platform Version                                                            : %s\n\n", oclPlatformInfo);

		//print supporting device number
		printf("[3]\tTotal Number Of OpenCL Supported Device / Devices On This System                                 : %d\n\n", dev_count);

		// allocate memory to hold this device ids
		ocl_device_ids = (cl_device_id *)malloc(sizeof(cl_device_id) * dev_count);

		//get ids into allocated buffer
		clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, dev_count, ocl_device_ids, NULL);

		char ocl_dev_prop[1024];
		for(int i = 0; i < dev_count; i++)
		{
			printf("************************************************************GPU DEVICE GENERAL INFORMATION************************************************************\n\n");
			printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");
			printf("[4]\tGPU Device Number                                                                                 : %d\n\n", i);

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_NAME, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			printf("[5]\tGPU Device Name                                                                                   : %s\n\n", ocl_dev_prop);

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VENDOR, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			printf("[6]\tGPU Device Vendor                                                                                 : %s\n\n", ocl_dev_prop);

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VERSION, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			printf("[7]\tGPU Device Version                                                                                : %s\n\n", ocl_dev_prop);

			clGetDeviceInfo(ocl_device_ids[i], CL_DRIVER_VERSION, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			printf("[8]\tGPU Driver Version                                                                                : %s\n\n", ocl_dev_prop);			

			cl_uint clock_frequency;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
			printf("[9]\tGPU Device Clock Rate                                                                             : %u\n\n", clock_frequency);

			cl_bool error_correction_support;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
			printf("[10]\tGPU Device Error Correction Code ( ECC ) Support                                                  : %s\n\n", error_correction_support == CL_TRUE ? "Yes" : "No");

			printf("************************************************************GPU DEVICE MEMORY INFORMATION*************************************************************\n\n");
			printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");	
			
			cl_ulong mem_size;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
			printf("[11]\tGPU Device Global Memory                                                                         : %llu Bytes\n\n", mem_size);

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
			printf("[12]\tGPU Device Local Memory                                                                          : %llu Bytes\n\n", mem_size);
		
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
			printf("[13]\tGPU Device Constant Buffer Size                                                                  : %llu Bytes\n\n", mem_size);

			cl_ulong max_mem_alloc_size;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
			printf("[14]\tGPU Device Memory Allocation Size                                                                : %llu Bytes\n\n", max_mem_alloc_size);

			printf("************************************************************GPU DEVICE COMPUTE INFORMATION************************************************************\n\n");
			printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");

			cl_uint compute_units;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
			printf("[15]\tGPU Device Number Of Parallel Processor Cores                                                    : %u\n\n", compute_units);

			size_t workGroup_size;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroup_size), &workGroup_size, NULL);
			printf("[16]\tGPU Device Work Group Size                                                                       : %u\n\n", (unsigned int)workGroup_size);

			size_t worktime_dims;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(worktime_dims), &worktime_dims, NULL);
			printf("[17]\tGPU Device Work Item Dimensions                                                                  : %u\n\n", (unsigned int)worktime_dims);

			size_t worktime_size[3];
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(worktime_size), &worktime_size, NULL);
			printf("[18]\tGPU Device Work Item Sizes                                                                       : %u %u %u\n\n", (unsigned int)worktime_size[0], (unsigned int)worktime_size[1], (unsigned int)worktime_size[2]);

			printf("***************************************************************GPU DEVICE IMAGE SUPPORT***************************************************************\n\n");
			printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");

			size_t szMaxDims[5];
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
			printf("[19]\tGPU Device Supported 2-D Image Width x Height                                                    : %u x %u\n\n", (unsigned int)szMaxDims[0], (unsigned int)szMaxDims[1]);

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);

			printf("[20]\tGPU Device Supported 3-D Image Width x Height x Depth                                            : %u x %u x %u\n\n", (unsigned int)szMaxDims[2], (unsigned int)szMaxDims[3], (unsigned int)szMaxDims[4]);

		}	
		free(ocl_device_ids);
	}
}
