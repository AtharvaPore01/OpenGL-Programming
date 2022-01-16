// OpenCL Kernel

__kernel void vecAdd(__global float *in1, __global float *in2, __global float *out, int len)
{
	//variable declaration
	int i = get_global_id(0);

	//code
	if(i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}