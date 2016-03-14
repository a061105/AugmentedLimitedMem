#ifndef UTIL
#define UTIL

#include<map>
#include<vector>
#include<string>
#include<cmath>
#include <iomanip>      // std::setprecision
#include <fstream>
#include <set>

using namespace std;

void split(string str, string pattern, vector<string>& tokens);

double maximum(double* values, int size);

double maximum(double* values, int size,int &posi);

int expOverSumExp(double *values, double *prob, int size);

double logSumExp(double* values, int size);

double normalize(double* values, int size);

void dataToFeatures( vector<vector<pair<int,double> > >& data, int dim,  //intput
		vector<vector<pair<int,double> > >& features //output
        );	

void softThd(double* w, vector<int> &act_set, double t_lambda);
void softThd(double* w, int size, double t_lambda);
double softThd(const double &x,const double  &thd);

double l1_norm(double* w, int size);

double l1_norm(vector<double>& w);

double l2_norm(double* w,int size);
void shuffle(vector<int>& arr);

double sign(double v);

typedef vector<pair<int,double> > Feature;


#endif
