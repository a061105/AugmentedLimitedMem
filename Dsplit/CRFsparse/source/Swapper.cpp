#include <fstream>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <map>

#include "util.h"
#include "Swapper.h"

using namespace std;

void writeArr(char* fname, ValueMap arr, int size){
	
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

void readArr(char* fname, ValueMap arr, int size){
	
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
void Swapper::save(int k, ValueMap arr, int size, char* name){
	
	//Swap current block into disk
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name,  k);
	writeArr( fname_arr[k], arr, size );
}

void Swapper::load(int k, ValueMap arr, int size, char* name){
	
	//Swap k_new block into memory
	sprintf(fname_arr[k], "%s/%s.%d", tmp_model_dir, name, k);
	readArr( fname_arr[k], arr, size );
}
