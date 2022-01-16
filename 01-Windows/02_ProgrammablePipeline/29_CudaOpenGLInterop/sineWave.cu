__global__ void sineWaveVBOKernel(float4 *pPos, unsigned int width, unsigned int height, float fAnimationTime)
{
	//code
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	u = (u * 2.0) - 1.0;
	v = (v * 2.0) - 1.0;

	float frequency = 4.0;
	float w = sinf(frequency * u + fAnimationTime) * cosf(frequency * v + fAnimationTime) * 0.5f;

	pPos[y * width + x] = make_float4(u, w, v, 1.0);
}

extern "C" void launchCudaKernel(float4 *pPos, unsigned int width, unsigned int height, float fAnimationTime)
{
	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	sineWaveVBOKernel << <grid, block >> > (pPos, width, height, fAnimationTime);
}
