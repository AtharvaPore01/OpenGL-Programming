//opencl kernel

__kernel void matrixMultiplication(__global int *A, __global int *B, __global int *C, int numARow, int numAColumns, int numBRow, int numBColumns, int numCRow, int numCColumns)
{
	//variable declaration
	int row = get_global_id(0);
	int col = get_global_id(1);

	//code
	if((row < numARow) && (col < numBColumns))
	{
		int CValue = 0;

		for(int k = 0; k < numAColumns; k++)
		{
			CValue += A[row * numAColumns + k] * B[k * numBColumns + col];
		}
		C[row * numCColumns + col] = CValue;
	}
}
