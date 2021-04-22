#include "gfcalu.cuh"
#include <stdio.h>

__device__
const int mgfaddtable[4][4] = {{0,1,2,3},{1,0,3,2},{2,3,0,1},{3,2,1,0}};
__device__
const int mgfsubtable[4][4] = {{0,1,2,3},{1,0,3,2},{2,3,0,1},{3,2,1,0}};
__device__
const int mgfmultable[4][4] = {{0,0,0,0},{0,1,2,3},{0,2,3,1},{0,3,1,2}};
__device__
const int mgfdivtable[4][4] = {{0,0,0,0},{0,1,3,2},{0,2,1,3},{0,3,2,1}};

__device__
int mgfadd(int a, int b)
{
	return mgfaddtable[a][b];
}
__device__
int mgfsub(int a, int b)
{
	return mgfsubtable[a][b];
}
__device__
int mgfmul(int a, int b)
{
	return mgfmultable[a][b];
}
__device__
int mgfdiv(int a, int b)
{
	return mgfdivtable[a][b];
}

__global__
void gfmatrixmulKernel(int *d_c,int *d_H, int *d_flag, int *d_m2n, int N, int M)
/*H和c展开成一维向量
矩阵是N行M列*/
{
    int tid = threadIdx.x + blockIdx.x* blockDim.x; //每个线程计算一行的结果
    int sum = 0;
    if(tid < N)
    {
        for (int i = 0; i < 5; i ++)
        {
            sum = mgfadd(sum,mgfmul(d_c[d_m2n[tid*5 + i]-1],d_H[tid*M + d_m2n[tid*5 + i]-1]));
        }
        d_flag[tid] = sum;
    }
}