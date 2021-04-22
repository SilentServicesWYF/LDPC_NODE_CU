#include "llrv.cuh"
#include "gfcalu.cuh"
#include <iostream>

//读显存专用函数
__global__
void testKernel(float *pointerArray, int row, int col)
{
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	if (c < col && r < row)
	{
		printf("(%d,%d):%f\n",r+1,c+1,pointerArray[r*col+c]);
	}
}

__global__
void estcKernel(float *d_Lpost, int *d_est_c)
// 最大似然解码的函数,可以一维block 每block32个线程同时做完
{
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n < 2688)
	{
		float temp = d_Lpost[n*4];
		int index = 0;
		for (int k = 1; k < 4; k ++)
		{
			if (d_Lpost[n*4+k] > temp)
			{
				temp = d_Lpost[n*4+k];
				index = k;
			}
		}
		d_est_c[n] = index;
	}
}

__device__
void Lpostupdate(float *d_Lpost, float *d_Lm2n, int targetm2n, int n_index, int m_index)
{
    for (int k = 0; k < 4; k ++)
    {
        float targetLm2n = d_Lm2n[m_index*5*4 + targetm2n*4 + k];
        d_Lpost[n_index*4 + k] = d_Lpost[n_index*4 + k] + targetLm2n;
    }
}

__global__
void LpostupdateKernel(float *d_Lpost, float *d_Lm2n, int *d_m2n, int *d_m2n_num, int *d_n2m, int *d_n2m_num)
{
    int n_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (n_index < 2688)
    {
        int mset_num = d_n2m_num[n_index];
        int mset[5];
        for (int k = 0; k < mset_num; k ++)
        {
            mset[k] = d_n2m[n_index*3 + k];
        }
        //搜索到所有与n连接的m节点之后对每一个被连接的m节点搜索其连接的n节点在Lm2n中的位置然后更新Lpost
        for (int k = 0; k < mset_num; k ++)
        {
            //搜索mset[k]对应的m2n的index
            int targetm2n = 0;
            for (int s = (mset[k]-1)*5; s < mset[k]*5; s ++)
            {
                if ((n_index + 1) == d_m2n[s])
                {
                    break;
                }
                targetm2n = targetm2n + 1;
            }
            Lpostupdate(d_Lpost, d_Lm2n, targetm2n, n_index, mset[k]-1);
        }
    }
}

__device__
void Ln2mupdate(float *d_Ln2m, float *d_Lm2n, int targetn2m, int targetm2n, int n_index, int m_index)
{
    for (int k = 0; k < 4; k ++)
    {
        float targetLm2n = d_Lm2n[m_index*5*4 + targetm2n*4 + k];
        d_Ln2m[n_index*3*4 + targetn2m*4 + k] = d_Ln2m[n_index*3*4 + targetn2m*4 + k] + targetLm2n;
    }
}

__global__
void Ln2mupdateKernel(float *d_Ln2m, float *d_Lm2n, int *d_n2m, int *d_m2n, int *d_n2m_num, int *d_m2n_num)
{
    int n_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (n_index < 2688)
    {
        int mset_num = d_n2m_num[n_index];
        int mset[5];
        for (int k = 0; k < mset_num; k++)
        {
            mset[k] = d_n2m[n_index*3 + k];
        }
        //对每个n连接的m节点计算除了本m以外连接到n的m节点进行LLRV更新
        for (int k = 0; k < mset_num; k ++)
        {
            //搜索除了本m节点外的m节点
            int msubset[4];
            int msel = mset[k];
            int subset_index = 0;
            for (int kk = 0; kk < mset_num; kk ++)
            {
                if (msel != mset[kk])
                {
                    msubset[subset_index] = mset[kk];
                }
                subset_index = subset_index + 1;
            }
            for (int kk = 0; kk < mset_num - 1; kk ++)
            {
                //搜索msubset[kk]对应的m2n的index
                int targetm2n = 0;
                for (int s = (msubset[kk]-1)*5; s < msubset[kk]*5; s ++)
                {
                    if ((n_index + 1) == d_m2n[s])
                    {
                        break;
                    }
                    targetm2n = targetm2n + 1;
                }
                Ln2mupdate(d_Ln2m, d_Lm2n, k, targetm2n, n_index, msubset[kk]-1);
            }
        }
    }
}