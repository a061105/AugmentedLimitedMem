#include<stdlib.h>
#include<vector>
#include<string>
#include<cmath>
#include "util.h"
/*
ValueMap createValueMap(Int cap){
	return map<Int,double>();
}

ArrMap createArrMap(Int cap){
	return map<Int,double*>();
}
*/
ValueMap createValueMap(Int cap){
	double* arr = new double[cap];
	for(Int i=0;i<cap;i++)
		arr[i] = 0.0;
	return arr;
}

ArrMap createArrMap(Int cap){
	return new double*[cap];
}


void split(string str, string pattern, vector<string>& tokens){

	tokens.clear();

	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		tokens.push_back(str.substr(i,index-i));

		i = index+1;
	}

	if( tokens.back()=="" )
		tokens.pop_back();
}

double softThd(const double &x,const double  &thd){
	if (x>thd)
		return x-thd;
	else if (x < -thd)
		return x+thd;
	else
		return 0;
}


double l2_norm(double* w, Int size){
    double norm=0.0;
    for (Int i=0;i<size;i++)
        norm+= w[i] * w[i];
    return norm/2.0;
}
double maximum(double* values, Int size,Int &posi){
	
	double ret = -1e300;
	for(Int i=0;i<size;i++){
        if( values[i] > ret ){
			ret = values[i];
            posi = i;
		}
	}
	return ret;
}

Int expOverSumExp(double *values, double *prob, Int size){
    Int posi = -1;
    double max_log = maximum(values,size,posi);
    double sum = 0.0;
    for (Int i=0;i<size;i++){
        prob[i] = exp(values[i]-max_log);
    }
    for (Int i=0;i<size;i++){
        sum += prob[i];
    }
    for (Int i=0;i<size;i++){
        prob[i] /= sum;
    }
    return posi;
}



double maximum(double* values, Int size){
	
	double ret = -1e300;
	for(Int i=0;i<size;i++){
		if( values[i] > ret ){
			ret = values[i];
		}
	}
	
	return ret;
}

double logSumExp(double* values, Int size){
	
	double sum=0.0;
	double max_log = maximum(values, size);
	for(Int i=0;i<size;i++){
		sum += exp(values[i] - max_log);
	}
	
	return log(sum) + max_log;
}



double normalize(double* values, Int size){
	
	double sum = 0.0;
	for(Int i=0;i<size;i++){
		sum += values[i];
	}
	for(Int i=0;i<size;i++){
		values[i] /= sum;
	}
	
	return log(sum);
}

void dataToFeatures( vector<vector<pair<Int,double> > >& data, Int dim,  //Intput
		vector<vector<pair<Int,double> > >& features //output
		){

	features.clear();
	for(Int i=0;i<dim;i++)
		features.push_back(vector<pair<Int,double> >());

	for(Int i=0;i<data.size();i++){

	    vector<pair<Int,double> >* ins = &(data[i]);

		vector<pair<Int,double> >::iterator it;
		for(it=ins->begin();it!=ins->end();it++){
			features[it->first].push_back(make_pair(i,it->second));
		}
	}

	//  for (Int i=0;i<data.size();i++)
	//      features[dim].push_back(make_pair(i,1.0));
}

void softThd(double* w, vector<Int> &act_set, double t_lambda){
	
	double abs_w ;
    Int i;
	for(Int ii=0; ii<act_set.size();ii++){
		i = act_set[ii];
		abs_w = fabs(w[i]);
		if( abs_w <= t_lambda ){
			
			w[i] = 0.0;

		}else if( w[i] > 0 ){
			
			w[i] = w[i] - t_lambda;
		}else{
			
			w[i] = w[i] + t_lambda;
		}
	}
}


void softThd(double* w, Int size, double t_lambda){
	
	double abs_w ;
	for(Int i=0; i<size;i++){
		
		abs_w = fabs(w[i]);
		if( abs_w <= t_lambda ){
			
			w[i] = 0.0;

		}else if( w[i] > 0 ){
			
			w[i] = w[i] - t_lambda;
		}else{
			
			w[i] = w[i] + t_lambda;
		}
	}
}

double l1_norm(double* w, Int size){
	
	double sum = 0.0;
	for(Int j=0;j<size;j++){
		sum += fabs(w[j]);
	}
	return sum;
}

double l1_norm(vector<double>& w){
	
	double sum = 0.0;
	for(Int j=0;j<w.size();j++){
		sum += fabs(w[j]);
	}
	return sum;
}

double l1_norm(double *w, vector<Int> &actset){
	
    double sum = 0.0;
	
    for (vector<Int>::iterator ii=actset.begin();ii != actset.end();++ii)
		sum += fabs(w[*ii]);
	
    return sum;
}

double l2_norm_sq(double *w, vector<Int> &actset){
	
    double sum = 0.0;
    for (vector<Int>::iterator ii=actset.begin();ii != actset.end();++ii)
		sum += w[*ii]*w[*ii];
    
    return sum;
}

void shuffle(vector<Int>& arr){
	
	Int size = arr.size();
	for(Int j=0; j<size; j++)
	{
		Int i = j+rand()%(size-j);
		swap(arr[i], arr[j]);
	}
}

double sign(double v){
	
	if( v > 0.0 )
		return 1.0;
	else if( v < 0.0 )
		return -1.0;
	else
		return 0.0;
}

void writeModel( const char* fname, double* w, Int d, Int raw_d){
    ofstream fout(fname);
    fout << "number of raw features:" << raw_d << endl;

    for(Int i=0;i<d;i++){
        if (w[i] != 0.0)
        fout << i << ":" << w[i]<<" ";
    }
    fout<<endl;
    fout.close();
}
