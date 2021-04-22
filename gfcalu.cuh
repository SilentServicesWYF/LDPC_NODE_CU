#ifndef GFCALU_CUH
#define GFCALU_CUH

__device__
int mgfadd(int a, int b);
__device__
int mgfsub(int a, int b);
__device__
int mgfmul(int a, int b);
__device__
int mgfdiv(int a, int b);
__global__
void gfmatrixmulKernel(int *d_c,int *d_H, int *d_flag, int *d_m2n, int N, int M);

#endif