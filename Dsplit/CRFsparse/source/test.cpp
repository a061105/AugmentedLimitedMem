#include <map>
#include <iostream>
#include <limits.h>
using namespace std;

typedef long Int;

typedef map<Int,double> ValueMap;
ValueMap createValueMap(Int cap){
	return *(new map<Int,double>());
}

/*
typedef map<Int,double*> ArrMap;
ArrMap& createArrMap(Int cap){
	return *(new map<Int,double*>());
}
*/

/*typedef double* ValueMap;
ValueMap createValueMap(Int cap){
	return new double[cap];
}
*/

class TestClass
{
	public:
	ValueMap v;
	
	TestClass(){
		int m = 10000;
		int n = 10000;
		v = createValueMap(m*n);

		cerr << "init" << endl;
		for(int i=0;i<m;i++)
			for(int j=0;j<n/10;j++)
				v[i*n+j]  = 1.0;
		cerr << "end init" << endl;
	}
};

int main(){
	
	TestClass* t = new TestClass();
	ValueMap& v2 = t->v;
	for(Int i=0;i<10;i++)
		v2[i] = 2.0;
	
	for(Int i=0;i<20;i++){
		cerr << t->v[i] << " ";
	}
	cerr << endl;

	cerr << "done" << endl;
}
