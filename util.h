#ifndef UTILITY
#define UTILITY

#include<stdlib.h>
#include<vector>
#include<string>
using namespace std;

const int PATH_LENGTH = 1000;
typedef vector<pair<int,double> > Instance;

vector<string> split(string str, string pattern);

void vectToArr( vector<string>& vect, int& size, char**& arr );

void shuffle(int*& nodelist, int size);
void shuffle(vector<int>& nodelist, int size);

double norm2_sq(double* a, double* b, int size);

#endif
