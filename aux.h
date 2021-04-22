#ifndef AUX_H
#define AUX_H

void readvector(int row, int *vector, char const *filename);
void readconstell(int row, float *constell, char const *filename);
float LLRV(float *subconstell, const int *pskdict, int gf);
float *floatslice(float *arry, int start_index, int end_index);
int divup(int a,int b);

#endif