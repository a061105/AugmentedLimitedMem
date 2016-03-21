#ifndef UTIL
#define UTIL

#include<vector>
#include<set>
#include<map>
#include<string>
#include<cmath>
#include <iomanip>      // std::setprecision
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

typedef int Int;

/*
typedef map<Int,double> ValueMap;
ValueMap createValueMap(Int cap);

typedef map<Int,double*> ArrMap;
ArrMap createArrMap(Int cap);
*/
typedef double* ValueMap;
ValueMap createValueMap(Int cap);

typedef double** ArrMap;
ArrMap createArrMap(Int cap);



const Int MAX_LINE = 500000000; //max char in a line in a file
const Int FNAME_LENGTH = 10000;//max length of a file name

class ScoreComp{
	
	public:
	double* score_arr;
	ScoreComp(double* _score_arr){
		score_arr = _score_arr;
	}
	bool operator()(const Int& var1, const Int& var2){
		return score_arr[var1] > score_arr[var2];
	}
};

typedef vector<pair<Int,double> > Feature;

void split(string str, string pattern, vector<string>& tokens);

double maximum(double* values, Int size);

double maximum(double* values, Int size,Int &posi);

Int expOverSumExp(double *values, double *prob, Int size);

double logSumExp(double* values, Int size);

double normalize(double* values, Int size);

void dataToFeatures( vector<vector<pair<Int,double> > >& data, Int dim,  //Intput
		vector<vector<pair<Int,double> > >& features //output
        );	

void softThd(double* w, vector<Int> &act_set, double t_lambda);
void softThd(double* w, Int size, double t_lambda);
double softThd(const double &x,const double  &thd);

double l1_norm(double* w, Int size);

double l1_norm(vector<double>& w);

double l1_norm(double *w, vector<Int> &actset);
double l2_norm_sq(double *w, vector<Int> &actset);

double l2_norm(double* w,Int size);
void shuffle(vector<Int>& arr);

double sign(double v);

void writeModel( const char* fname, double* w, Int d, Int raw_d);

#endif
