#include <fstream>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include "Swapper.h"

using namespace std;

void writeArr(char* fname, double* arr, int size){
	
	FILE* pFile;
	pFile = fopen(fname,"wb");
	
	vector<int> nzIndex;
	for(int i=0;i<size;i++){
		if( arr[i] != 0.0 ){
			nzIndex.push_back(i);
		}
	}
	
	int nnz = nzIndex.size();
	fwrite( &nnz, sizeof(int), 1, pFile);
	for(int i=0;i<nnz;i++){
		int r= nzIndex[i];
		fwrite( &r, sizeof(int), 1, pFile);
		fwrite( &(arr[r]), sizeof(double), 1, pFile);
	}
	fclose(pFile);
}

void readArr(char* fname, double* arr, int size){
	
	FILE* pFile;
	pFile = fopen(fname,"rb");
	if( pFile == NULL ){
		for(int i=0;i<size;i++)
			arr[i] = 0.0;
		return ;
	}

	for(int i=0;i<size;i++){
		arr[i] = 0.0;
	}
	
	int nnz,r;
	double v;
	fread(&nnz,sizeof(int),1,pFile);
	for(int i=0;i<nnz;i++){
		fread(&r,sizeof(int),1,pFile);
		fread(&v,sizeof(double),1,pFile);
		arr[r] = v;
	}
	fclose(pFile);
}
void Swapper::save(int k, double* arr, int size, char* name){
	
	//Swap current block into disk
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name,  k);
	writeArr( fname_arr[k], arr, size );
}

void Swapper::load(int k, double* arr, int size, char* name){
	
	//Swap k_new block into memory
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name, k);
	readArr( fname_arr[k], arr, size );
}
void writeArr(char* fname, double* arr, long long size){
	
	FILE* pFile;
	pFile = fopen(fname,"wb");
	
	vector<long long> nzIndex;
	for(long long i=0;i<size;i++){
		if( arr[i] != 0.0 ){
			nzIndex.push_back(i);
		}
	}
	
	long long nnz = nzIndex.size();
	fwrite( &nnz, sizeof(long long), 1, pFile);
	for(int i=0;i<nnz;i++){
		long long r= nzIndex[i];
		fwrite( &r, sizeof(long long), 1, pFile);
		fwrite( &(arr[r]), sizeof(double), 1, pFile);
	}
	fclose(pFile);
}

void readArr(char* fname, double* arr, long long size){
	
	FILE* pFile;
	pFile = fopen(fname,"rb");
	if( pFile == NULL ){
		for(long long i=0;i<size;i++)
			arr[i] = 0.0;
		return ;
	}

	for(long long i=0;i<size;i++){
		arr[i] = 0.0;
	}
	
	long long nnz,r;
	double v;
	fread(&nnz,sizeof(long long),1,pFile);
	for(int i=0;i<nnz;i++){
		fread(&r,sizeof(long long),1,pFile);
		fread(&v,sizeof(double),1,pFile);
		arr[r] = v;
	}
	fclose(pFile);
}
void Swapper::save(int k, double* arr, long long size, char* name){
	
	//Swap current block into disk
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name,  k);
	writeArr( fname_arr[k], arr, size );
}

void Swapper::load(int k, double* arr, long long size, char* name){
	
	//Swap k_new block into memory
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name, k);
	readArr( fname_arr[k], arr, size );
}
void writeArr(char* fname, int* arr, int size){
	
	FILE* pFile;
	pFile = fopen(fname,"wb");
	
	vector<int> nzIndex;
	for(int i=0;i<size;i++){
		if( arr[i] != 0 ){
			nzIndex.push_back(i);
		}
	}
	
	int nnz = nzIndex.size();
	fwrite( &nnz, sizeof(int), 1, pFile);
	for(int i=0;i<nnz;i++){
		int r= nzIndex[i];
		fwrite( &r, sizeof(int), 1, pFile);
		fwrite( &(arr[r]), sizeof(int), 1, pFile);
	}
	fclose(pFile);
}

void readArr(char* fname, int* arr, int size){
	
	FILE* pFile;
	pFile = fopen(fname,"rb");
	if( pFile == NULL ){
		for(int i=0;i<size;i++)
			arr[i] = 0;
		return ;
	}
	
	int nnz,r;

	for(int i=0;i<size;i++){
		arr[i] = 0;
	}
	
	fread(&nnz,sizeof(int),1,pFile);
	int vi;
	for(int i=0;i<nnz;i++){
		fread(&r,sizeof(int),1,pFile);
		fread(&vi,sizeof(int),1,pFile);
		arr[r] = vi;
	}
	fclose(pFile);
}
void Swapper::save(int k, int* arr, int size, char* name){
	
	//Swap current block into disk
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name,  k);
	writeArr( fname_arr[k], arr, size );
}

void Swapper::load(int k, int* arr, int size, char* name){
	
	//Swap k_new block into memory
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name, k);
	readArr( fname_arr[k], arr, size);
}
