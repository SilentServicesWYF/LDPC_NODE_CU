#include "aux.h"
#include <fstream>
#include <string.h>
#include <iostream>

using namespace std;

void readvector(int row, int *vector, char const *filename)
{
    ifstream fin;
    fin.open(filename);
    for (int i = 0; i < row; i++)
    {
        fin >> vector[i];
    }
    cout<<filename<<"导入完成"<<endl;
}

void readconstell(int row, float *constell, char const *filename)
{
    ifstream fin;
    fin.open(filename);
    for (int i = 0; i < row; i++)
    {
        fin >> constell[i];
    }
    cout<<filename<<"导入完成"<<endl;
}

float LLRV(float *subconstell, const int *pskdict, int gf)
/*对输入的星座点计算对应的gf元素的对数似然比函数*/
{
    float llrv = 0;
    int index_start = gf*2;
    int index_end = gf*2+1;
    int x_index = 0;
    for (int i = index_start; i <= index_end; i++)
    {
        if(pskdict[i]==1)
        {
            llrv = llrv + subconstell[x_index];
        }
        x_index = x_index + 1;
    }
    return llrv;
}

float *floatslice(float *arry, int start_index, int end_index)
/*用于float类型的切片*/
{
    float *a = new float[end_index-start_index + 1];
    int count = 0;
    for (int i = start_index; i <= end_index;i++)
    {
        a[count] = arry[i];
        count = count + 1;
    }
    return a;
}

int divup(int a,int b)
{
    if(a%b==0){
        return a/b;
    }
    else{
        return a/b+1;
    }
}