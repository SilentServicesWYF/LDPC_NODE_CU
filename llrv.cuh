#ifndef LLRV_CUH
#define LLRV_CUH

__global__
void testKernel(float *pointerArray, int row, int col);
__global__
void estcKernel(float *d_Lpost, int *d_est_c);
__device__
void Lpostupdate(float *d_Lpost, float *d_Lm2n, int targetm2n, int n_index, int m_index);
__global__
void LpostupdateKernel(float *d_Lpost, float *d_Lm2n, int *d_m2n, int *d_m2n_num, int *d_n2m, int *d_n2m_num);
__device__
void Ln2mupdate(float *d_Ln2m, float *d_Lm2n, int targetn2m, int targetm2n, int n_index, int m_index);
__global__
void Ln2mupdateKernel(float *d_Ln2m, float *d_Lm2n, int *d_n2m, int *d_m2n, int *d_n2m_num, int *d_m2n_num);
#endif