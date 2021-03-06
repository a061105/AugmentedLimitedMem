#include <fstream>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <map>

const int PATH_LENGTH = 1000;

class Swapper{
	public:
		int K;
		char* tmp_model_dir;
		char** fname_arr;
		
		Swapper(int numBlock, bool mkdir){
			
			K = numBlock;
			//crate tmp model directory
			tmp_model_dir = new char[PATH_LENGTH];
			time_t now;
			time(&now);
			sprintf(tmp_model_dir,"tmp_info.%d",now);
			char tmp_cmd[PATH_LENGTH];
			if(mkdir == true)
			{
				sprintf(tmp_cmd,"mkdir %s",tmp_model_dir);
				system(tmp_cmd);
			}

			fname_arr = new char*[K];
			
			for(int i=0;i<K;i++){
				
				fname_arr[i] = new char[PATH_LENGTH];
				
				sprintf(fname_arr[i], "%s/", tmp_model_dir);
			}
		}

		
		~Swapper(){
			
			char tmp[1000];
			sprintf(tmp,"rm -rf %s",tmp_model_dir);
			system(tmp);
			
			for(int i=0;i<K;i++){
				delete[] fname_arr[i];
			}
			delete[] fname_arr;
		}
		
		void save(int k, double* arr, int size, char* name);
		void load(int k, double* arr, int size, char* name);
		void save(int k, double* arr, long long size, char* name);
		void load(int k, double* arr, long long size, char* name);
		
		void load(int k, int* arr, int size, char* name);
		void save(int k, int* arr, int size, char* name);

		int num_block(){
			return K;
		}
};


