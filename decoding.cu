#include "aux.h"
#include "gfcalu.cuh"
#include "llrv.cuh"
#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <time.h>
#include <sys/time.h>

//参数设定
int row1 = 1344;
int col1 = 2688;
int row2 = 2688;
int col2 = 1;
const int maxweight1 = 3;
const int maxweight2 = 5;
const int pskdict[8] = {-1,-1,-1,1,1,-1,1,1};

int main()
{
	// 初始化内存
	int *H = new int [row1*col1];
	int *c = new int [col1];
	int *m2n = new int [row1*maxweight2];
	int *n2m = new int [row2*maxweight1];
	float *constell = new float [col1*2];
	int *n2m_num = new int [col1];
	int *m2n_num = new int [row1];

	readvector(col1,n2m_num,"data/n2m_num.txt");
	readvector(row1,m2n_num,"data/m2n_num.txt");
	readvector(row1*col1,H,"data/H.txt");
	readvector(col1,c,"data/c.txt");
	readvector(row1*maxweight2,m2n,"data/m2n.txt");
	readvector(row2*maxweight1,n2m,"data/n2m.txt");
	readconstell(col1*2,constell,"data/constell.txt");

	/*复制m2n,n2m,m2n_num,n2m_num,H到显存*/
	int *d_m2n = 0;
	int *d_n2m = 0;
	int *d_m2n_num = 0;
	int *d_n2m_num = 0;
	int *d_H = 0;

	cudaMalloc(&d_m2n, sizeof(int)*row1*maxweight2);
	cudaMalloc(&d_n2m, sizeof(int)*row2*maxweight1);
	cudaMalloc(&d_m2n_num, sizeof(int)*row1);
	cudaMalloc(&d_n2m_num, sizeof(int)*col1);
	cudaMalloc(&d_H, sizeof(int)*row1*col1);

	cudaMemcpy(d_m2n, m2n, sizeof(int)*row1*maxweight2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n2m, n2m, sizeof(int)*row2*maxweight1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2n_num, m2n_num, sizeof(int)*row1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n2m_num, n2m_num, sizeof(int)*col1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_H, H, sizeof(int)*col1*row1, cudaMemcpyHostToDevice);

	//主机端计算Lch
	float *subconstell;
	float *Lch = new float [col1*4];
	for (int n = 0; n < col1; n++)
	{
		subconstell =  floatslice(constell, n*2, n*2+1);
		for (int i = 0; i < 4; i ++)
		{
			Lch[n*4+i] = LLRV(subconstell, pskdict, i);
		}
		delete []subconstell;
	}
	//把Lch放到显存中
	float *d_Lch = 0;
	cudaMalloc(&d_Lch, col1*4*sizeof(float));
	cudaMemcpy(d_Lch, Lch, col1*4*sizeof(float), cudaMemcpyHostToDevice);

	/*在主机端初始化Ln2m,Lm2n,Ln2mbuff*/
	float *Ln2m = new float [row2*maxweight1*4]();
	float *Lm2n = new float [row1*maxweight2*4]();
	float *Ln2mbuff = new float [row2*maxweight1*4]();
	for (int k = 0; k < row2; k ++)
	{
		int avm_index = 0;
		while ((n2m[k*maxweight1+avm_index] != 0) && (avm_index < maxweight1))
		{
			int Lch_index = 0;
			for (int buff_index = avm_index*4; buff_index < (avm_index + 1)*4; buff_index ++)
			{
				Ln2mbuff[k*maxweight1*4+buff_index] = Lch[k*4+Lch_index];
				Lch_index = Lch_index + 1;
			}
			avm_index = avm_index + 1;
		}
	}
	/*复制Ln2m,Lm2n,Ln2mbuff到显存*/
	float *d_Ln2m = 0;
	float *d_Lm2n = 0;
	float *d_Ln2mbuff = 0;
	cudaMalloc(&d_Ln2m, sizeof(float)*row2*maxweight1*4);
	cudaMalloc(&d_Lm2n, sizeof(float)*row1*maxweight2*4);
	cudaMalloc(&d_Ln2mbuff, sizeof(float)*row2*maxweight1*4);
	cudaMemcpy(d_Ln2m, Ln2m, sizeof(float)*row2*maxweight1*4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Lm2n, Lm2n, sizeof(float)*row1*maxweight2*4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ln2mbuff, Ln2mbuff, sizeof(float)*row2*maxweight1*4, cudaMemcpyHostToDevice);

	//迭代参数
	int iterflag = 1;
	int iter_num = 0;
	int maxiter = 1;
	//为显存中的Lpost和est_c申请空间
	float *d_Lpost = 0; //每次赋值的时候直接从内存拷贝
	int *d_est_c = 0;
	int *h_est_c = new int[col1](); //矩阵相乘在设备端调用
	int *h_flag = new int [row1](); //矩阵相乘在设备端返回的结果
	int *d_flag = 0;
	int *d_c = 0;

	cudaMalloc(&d_Lpost, col1*4*sizeof(float));
	cudaMalloc(&d_est_c, col1*sizeof(int));
	cudaMalloc(&d_flag, row1*sizeof(int));
	cudaMalloc(&d_c, col1*sizeof(int));

	cudaMemcpy(d_est_c, h_est_c, col1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_flag, h_flag ,row1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, col1*sizeof(int), cudaMemcpyHostToDevice);

    while (iterflag == 1 && iter_num < maxiter)
    {
        iter_num = iter_num + 1;
        //为显存中的Lpost赋初值
		cudaMemcpy(d_Lpost, d_Lch, col1*4*sizeof(float), cudaMemcpyDeviceToDevice);
        // 更新Lpost
        int lblocksize = 32;
        int lgridsize = divup(col1, lblocksize);
        LpostupdateKernel<<<lgridsize, lblocksize>>>(d_Lpost, d_Lm2n, d_m2n, d_m2n_num, d_n2m, d_n2m_num);
        // dim3 blocksize(4,32);
        // dim3 gridsize(divup(4,4),divup(col1,32));
        // testKernel<<<gridsize,blocksize>>>(d_Lpost, col1, 4);
        // cudaDeviceSynchronize();
        // 根据Lpost计算最大似然译码
		int cblocksize = 32;
		int cgridsize = divup(col1, cblocksize);
		estcKernel<<<cgridsize, cblocksize>>>(d_Lpost, d_est_c);
        // 水平信息传递
		// 初始化Ln2m
		cudaMemcpy(d_Ln2m, d_Ln2mbuff, sizeof(float)*row2*maxweight1*4, cudaMemcpyDeviceToDevice);
        //更新Ln2m
        int hblocksize = 32;
        int hgridsize = divup(col1, hblocksize);
        Ln2mupdateKernel<<<hblocksize, hgridsize>>>(d_Ln2m, d_Lm2n, d_n2m, d_m2n, d_n2m_num, d_m2n_num);
        // dim3 blocksize(12,32);
		// dim3 gridsize(divup(12,12),divup(row1,32));
		// testKernel<<<gridsize,blocksize>>>(d_Ln2m, col1, 12);
		// cudaDeviceSynchronize();

        //垂直信息传递
        int vblocksize = 32;
        int vgridsize = divup(row1, vblocksize);
        Lm2nupdateKernel<<<vgridsize, vblocksize>>>(d_Lm2n, d_Ln2m, d_m2n, d_n2m, d_m2n_num, d_n2m_num, d_H);
        
    }
    return 0;
}