#include "util.h"

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}

	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

void vectToArr( vector<string>& vect, int& size, char**& arr ){ //deep copy
	
	size = vect.size();
	arr = new char*[size];
	for(int i=0;i<size;i++){
			
		char* cstr = new char[vect[i].size()+1];
		int j;
		for(j=0;j<vect[i].size();j++){
			cstr[j] = vect[i][j];
		}
		cstr[j] = '\0';
		
		arr[i] = cstr;
	}
}

template <class T> static inline void Swap(T& x, T& y) { T t=x; x=y; y=t; }

void shuffle(int*& nodelist, int size){
	
	for(int i = 0;i < size;i++){
		
		int j = i+rand()%(size-i);
		Swap(nodelist[i],nodelist[j]);
	}
}

void shuffle(vector<int>& nodelist, int size){
	
	for(int i = 0;i < size;i++){
		
		int j = i+rand()%(size-i);
		Swap(nodelist[i],nodelist[j]);
	}
}

double norm2_sq(double* a, double* b, int size){

	double sum = 0.0;
	double tmp;
	for(int i=0;i<size;i++){
		
		tmp = a[i] - b[i];
		sum += tmp*tmp;
	}
	
	return sum;
}

