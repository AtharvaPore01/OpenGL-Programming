//headers
#include<stdio.h>

int main(void)
{
	//function declaration
	void PrintCUDADeviceProperties(void);

	//code
	PrintCUDADeviceProperties();

	return(0);
}

void PrintCUDADeviceProperties(void)
{
	//function declaration
	int ConvertSMVersionNumberToCores(int, int);

	//code
	printf("\n\n");
	printf("***********************************************************************************************************************************************************************\n\n");
	printf("-------------------------------------------------------------------\n\n");
	printf("CUDA INFORMATION : \n\n");
	printf("-------------------------------------------------------------------\n");

	cudaError_t ret_cuda_rt;
	int dev_count;
	ret_cuda_rt = cudaGetDeviceCount(&dev_count);
	if(ret_cuda_rt != cudaSuccess)
	{
		printf("CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To %s.\nExitting Now....\n", cudaGetErrorString(ret_cuda_rt));
	}
	else if(dev_count == 0)
	{
		printf("There Is No CUDA Supported Device On This System.\n Exitting...\n");
		return;
	}
	else
	{
		printf("[1]\tTotal Number Of CUDA Supporting GPU Devices On This System : %d \n\n", dev_count);
		for(int i = 0; i < dev_count; i++)
		{
			cudaDeviceProp dev_prop;
			int driverVersion = 0, runtimeVersion = 0;

			ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, i);
			if(ret_cuda_rt != cudaSuccess)
			{
				printf("%s in %s At Line %d.\n", cudaGetErrorString(ret_cuda_rt), __FILE__, __LINE__);
				return;
			}
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			printf("********************* CUDA DRIVER AND RUNTIME INFORMATION *********************\n\n");
			printf("[2]\tCUDA Driver Version                                   : %d\n", driverVersion / 1000, (driverVersion % 100) / 10);
			printf("[3]\tCUDA Runtime Version                                  : %d\n\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

			printf("********************* GPU DEVICE GENERAL INFORMATION *********************\n\n");
			printf("[4]\tGPU Device Number                                     : %d\n\n", i);
			printf("[5]\tGPU Device Name                                       : %s\n\n", dev_prop.name);
			printf("[6]\tGPU Device Compute Capability                         : %d\n\n", dev_prop.major, dev_prop.minor);
			printf("[7]\tGPU Device Clock Rate                                 : %d\n\n", dev_prop.clockRate);
			printf("[8]\tGPU Device Type                                       : ");
			if(dev_prop.integrated)
			{
				printf("Integrated ( On-Board )\n\n");
			}
			else
			{
				printf("Discrete ( Card )\n\n");
			}

			printf("********************* GPU DEVICE MEMORY INFORMATION *********************\n\n");
			printf("[9]\tGPU Device Total Memory                               : %.0f GB = %.0f MB = %llu Bytes\n\n", ((float)dev_prop.totalGlobalMem / 1048576.0f) / 1024.0f, (float)dev_prop.totalGlobalMem / 1048576.0f, (unsigned long long)dev_prop.totalGlobalMem);
			printf("[10]\tGPU Device Available Memory                           : %lu Bytes\n\n", (unsigned long)dev_prop.totalConstMem);
			printf("[11]\tGPU Device Host Memory Mapping Capability             : ");
			if(dev_prop.canMapHostMemory)
			{
				printf("Yes ( Can Map Host Memory To Device Memory )\n\n");
			}
			else
			{
				printf("No ( Can Not Map Host Memory To Device Memory )\n\n");
			}

			printf("********************* GPU DEVICE MULTIPROCESSOR INFORMATION *********************\n\n");
			printf("[12]\tGPU Device Number Of SMProcessors                     : %d\n\n", dev_prop.multiProcessorCount);
			printf("[13]\tGPU Device Number Of Cores Per SMProcessor            : %d\n\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor));
			printf("[14]\tGPU Device Total Number Of Cores                      : %d\n\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
			printf("[15]\tGPU Device Shared Memory Per SMProcessor              : %lu\n\n", (unsigned long)dev_prop.sharedMemPerBlock);
			printf("[16]\tGPU Devoce Number Of Register Per SMProcessor         : %d\n\n", dev_prop.regsPerBlock);

			printf("********************* GPU DEVICE THREAD INFORMATION *********************\n\n");
			printf("[17]\tGPU Device Maximum Number Of Thread Per SMProcessor   : %d\n\n", dev_prop.maxThreadsPerMultiProcessor);
			printf("[18]\tGPU Device Maximum Number Of Threads Per Block        : %d\n\n", dev_prop.maxThreadsPerBlock);
			printf("[19]\tGPU Device Threads In Warp                            : %d\n\n", dev_prop.warpSize);
			printf("[20]\tGPU Device Maximum Thread Dimension                   : ( %d, %d, %d )\n\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);

			printf("********************* GPU DEVICE DRIVER INFORMATION *********************\n\n");
			printf("[21]\tGPU Device Has ECC Support                            : %s\n\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
			#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				printf("[22]\tGPU Device CUDA Driver Mode ( TCC or WDDM )           : %s\n\n", dev_prop.tccDriver ? "TCC ( Tesla Compute Cluster Driver )" : "WDDM ( Windows Display Driver Model )"); 
			#endif
			printf("***********************************************************************************************************************************************************************\n\n");
		}
	}
}

int ConvertSMVersionNumberToCores(int major, int minor)
{
	//defines for GPU Architecture types ( using the SM Version to determine the # of cores per SM )
	typedef struct
	{
		int SM; // 0xMm ( Hexadecimal Notation ), M = SM Major Version and m = SM Minot Version
		int Cores;
	}sSMtoCores;	

	sSMtoCores nGpuArchCoresPerSM[] = 
	{
		{ 0x20, 32 },
		{ 0x21, 48 },
		{ 0x30, 192 },
		{ 0x32, 192 },
		{ 0x35, 192 },
		{ 0x37, 192 },
		{ 0x50, 128 },
		{ 0x52, 128 },
		{ 0x53, 128 },
		{ 0x60, 64 },
		{ 0x61, 128 },
		{ 0x62, 128 },
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return(nGpuArchCoresPerSM[index].Cores);
		}
		index++;
	}
	//if we don't find the values, we default use the previous one to run properly
	printf("MapSMToCores for SM %d.%d is undefined. Default to use %d cores/SM\n\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return(nGpuArchCoresPerSM[index - 1].Cores);
}
