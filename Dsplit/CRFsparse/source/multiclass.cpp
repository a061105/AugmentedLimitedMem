#include <fstream>
#include <iostream>
#include <algorithm>
#include "util.h"
#include "multiclass.h"

using namespace std;


//Functions for MulticlassProblem only
MulticlassProblem::MulticlassProblem(char* data_file){

	K = 0;
	N = 0;
	d = raw_d = -1;
	readProblem(data_file);
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

MulticlassProblem::MulticlassProblem(char* model_file, char* data_file){

	K = 0;
	N = 0;
	d = raw_d = -1;
	readProblem(model_file, data_file);
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

//Functions for "Problem" Interface
void MulticlassProblem::compute_fv_change(double* w_change, vector<Int>& act_set, //input
		vector<pair<Int,double> >& fv_change){ //output

	double delta_w;
	Int w_index;
	Feature* feature;
	Feature::iterator it;
	
	//n = K*N fv
	double* fv_change_table = new double[n];
	for(Int i=0;i<n;i++)
		fv_change_table[i] = 0.0;
	
	Int i;
	Int fv_index;
	Int label;
	pair<Int,double>* pair;
	for(i=0; i<act_set.size();i++){

		w_index = act_set[i];
		delta_w = w_change[ w_index ];
		label = w_index % K;
		//find factors(w_index)
		feature = &(data_inv[ w_index/K ]);
		for(it=feature->begin();it!=feature->end();it++){
			
			fv_index = it->first + label;
			fv_change_table[fv_index] += delta_w * it->second;
		}
	}
	
	//copy to vector
	fv_change.clear();
	for(Int i=0;i<n;i++)
		if(fv_change_table[i] != 0.0){
			fv_change.push_back(make_pair(i,fv_change_table[i]));
		}

	delete[] fv_change_table;
}

void MulticlassProblem::update_fvalue(vector<pair<Int,double> >& fv_change, double scalar){
	
	sample_updated.clear();
	
	Int prev_factor = -1, factor;
	vector<pair<Int,double> >::iterator it;
	for(it=fv_change.begin();it!=fv_change.end(); it++){

		fvalue[it->first] += scalar * it->second;
		
		//record which samples have been updated
		factor = it->first/K;
		if( factor != prev_factor ){
			sample_updated.insert(factor);
		}
		prev_factor = factor;
	}
	
	inferMarg();
}

void MulticlassProblem::inferMarg(){

	logZ = 0.0;
	double logf_xy_max;
	for(Int i=0;i<N;i++){

		if( sample_updated.find(i) == sample_updated.end() ){
			continue;
		}
		double* f_xy = factor_xy[i];
		double* marg = marginal[i];
		
		logf_xy_max = maximum(f_xy,K);
		for(Int k=0;k<K;k++)
			marg[k] = exp(f_xy[k]-logf_xy_max);
		
		logZ += normalize(marg,K) + logf_xy_max;
	}
}

void MulticlassProblem::grad( vector<Int>& act_set, //input
		vector<double>& grad){ //output

	grad.clear();
	
	//for each active feature, collect residual at each factor (sample)
	Feature* feature;
	Int w_index, w_label, label;
	Int sample_index, fvalue_index;
	double g, loss_deriv;
	Int ind, fea_index;
	for(ind=0; ind<act_set.size(); ind++){
		
		w_index = act_set[ind];
		fea_index = w_index / K ;
		w_label = w_index % K;
		
		feature = &(data_inv[fea_index]);
		
		g = 0.0;
		vector<pair<Int,double> >::iterator it;
		for(it=feature->begin();it!=feature->end();it++){
			
			sample_index = (it->first/K);
			label = labels[sample_index];
			
			//loss derivative
			if( w_label == label ){
				loss_deriv = marginal[sample_index][w_label] - 1.0;
				//cerr << label_name_map.find(w_label)->second << ":" << marginal[factor_index][w_label]-1.0 << endl;
			}else
				loss_deriv = marginal[sample_index][w_label];

			//gradient
			g += loss_deriv * it->second;
		}
		
		g += param.theta*w[w_index]; //L2-regularization
		grad.push_back(g);
	}
}

double MulticlassProblem::fun(){

	double** f_xy;
	double nll = 0.0;
	for(Int i=0;i<N;i++){
		
		//log potential of i-th instance
		nll += -factor_xy[i][ labels[i] ];
	}
	nll += logZ;
	//cerr << logZ << " - " << log_pot_i << endl;
	
	double w_uni_norm_sq=0.0;
	for(Int j=0;j<d;j++)
		w_uni_norm_sq += w[j]*w[j];
	
	//cerr << logZ << " - " << log_pot_i << endl;
	
	return nll + param.theta*w_uni_norm_sq/2.0 ;
}

/** Given a w, compute all factor values.
 */
void MulticlassProblem::compute_fvalue(){
	
	for(Int i=0;i<n;i++){
		fvalue[i] = 0.0; 
	}
	//unigram factor
	Int findex, windex;
	for(Int i=0;i<N;i++){
		Feature* fea = data[i];
		for(Int k=0;k<K;k++){

			findex = i*K + k; 
			Feature::iterator it;
			//cerr << i << ", " << j << ", " << k << endl;
			for(it=fea->begin();it!=fea->end();it++){
				windex = it->first * K + k;
				if( windex >= d )
					continue;
				fvalue[findex] += w[ windex ]*it->second;
			}
		}
	}
	inferMarg();
}

double MulticlassProblem::train_accuracy(){
	double n_hit = 0.0;
	Int pred;
	for(Int i=0;i<N;i++){

		maximum(marginal[i], K, pred);
		n_hit += (pred == labels[i]) ;
	}

	return n_hit / N ;
}


void MulticlassProblem::test_accuracy(const char* output_file){

	ofstream fout(output_file);
	compute_fvalue();
	double accuracy = 0.0;
	Int pred;
	for(Int i=0;i<N;i++){
		maximum(marginal[i], K, pred);
		accuracy += (pred == labels[i]) ;
		
		fout << label_name_map[pred] <<endl; 
		fout<<endl;
	}
	cerr << "testing accuracy: " << accuracy / N <<endl;
	
	fout.close();
};


void MulticlassProblem::readProblem(char* model_file, char* data_file){

	char* _line = new char[MAX_LINE];
	vector<string> tokens;
	vector<string> iv_pair;

	//read raw dimension
	ifstream fin_model(model_file);
	fin_model.getline(_line, MAX_LINE);
	string line = string(_line);
	split( line, ":", tokens );
	raw_d = atoi(tokens[1].c_str());

	//read sequence data
	readProblem(data_file);
	
	//read model
	fin_model.getline(_line, MAX_LINE);
	line = string(_line);
	split( line, " ", tokens );
	for(Int i=0;i<tokens.size();i++){
		split(tokens[i], ":", iv_pair);
		Int ind = atoi(iv_pair[0].c_str());
		double val = atof(iv_pair[1].c_str());
		w[ind] = val;
	}
	fin_model.close();

	delete[] _line;
}

void MulticlassProblem::readProblem(char* data_file){

	char* _line = new char[MAX_LINE];
	string line;
	vector<string> tokens;
	vector<string> iv_pair;

	/*Read Feature Template
	*/
	if( param.info_file == NULL ){
		cerr << "exit: Multiclass Problem needs feature template specified as info file" << endl;
		exit(0);
	}
	ifstream finfo(param.info_file);
	if (finfo.fail()){
		cerr<< "can't open info file."<<endl;
		exit(0);
	}

	//filter one line ("label:")
	finfo.getline(_line,MAX_LINE); 
	finfo.getline(_line,MAX_LINE);
	line = string(_line);
	split(line," ", tokens);
	for(Int i=0;i<tokens.size();i++)
		label_index_map.insert( make_pair(tokens[i], (Int)label_index_map.size()) );
        K= label_index_map.size();
	for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++)
		label_name_map.insert(make_pair(it->second,it->first));

	finfo.close();
	
	/* Read samples from data file
	*/
	ifstream fin(data_file);
	if (fin.fail()){
		cerr<< "can't open data file."<<endl;
		exit(0);
	}
	Int count_line = 1;
	Int max_index = -1;
	while( !fin.eof() ){
		fin.getline(_line,MAX_LINE);
		
		string line(_line);
		if( line.size() <= 1 && !fin.eof() ){
			cerr << "data file line " << count_line << " format error." ;
			exit(0);
		}else if( line.size() <= 1 ){
			break;
		}

		split(line," ", tokens);
		
		//Get Label
		map<string,Int>::iterator it;
		if( (it=label_index_map.find(tokens[0])) != label_index_map.end() ){
			labels.push_back( it->second );
		}else{
			cerr << "data file line " << count_line << ": Unknown Label '" << tokens[0] << "'" << endl;
			exit(0);
		}
		
		//Get Features
		Feature* fea = new Feature();
		for(Int j=1;j<tokens.size();j++){
			split(tokens[j],":",iv_pair);
			Int index = atoi(iv_pair[0].c_str());
			double value =  atof(iv_pair[1].c_str());
			fea->push_back( make_pair( index, value ) );

			if(index > max_index)
				max_index = index;
		}
		
		data.push_back(fea);
		count_line++;
		//cerr << "line=" << count_line << endl;
	}
	fin.close();
	if(data.back()->size() == 0 ){
		data.pop_back();
		labels.pop_back();
	}
	
	N = data.size();
	if( raw_d == -1 ){ 
		raw_d = max_index;
	}
	d = (1+raw_d)*K;
	
	w = new double[d];
	for(Int i=0;i<d;i++)
		w[i] = 0.0;
	
	/*Build data inverted index
	*/
	build_data_inv();
	
	/*Build Factor Table Array
	*/
	n = N*K;
	
	fvalue = new double[n];
	for(Int i=0;i<n;i++)
		fvalue[i] = 0.0;

	Int k=0;
	factor_xy = new double*[N];
	for(Int i=0;i<N;i++){
		
		factor_xy[i] = &(fvalue[k]);
		k += K;
	}
	/*Build marginal array for each factor
	*/
	marginal = new double*[N];
	for(Int i=0;i<N;i++){
		marginal[i] = new double[K];
	}
	
	delete[] _line;
}

void MulticlassProblem::build_data_inv(){

	//initialize data_inv
	for(Int i=0;i<(d/K);i++)
		data_inv.push_back(Feature());
	
	//scan data, expand feature, add to data_inv
	Feature fea;
	Int fea_index;
	double fea_val;
	for(Int i=0;i<N;i++){
		Feature* fea = data[i];

		for(Feature::iterator it=fea->begin(); it!=fea->end(); it++){
			fea_index = it->first;
			fea_val = it->second;
			data_inv[fea_index].push_back( make_pair(i*K, fea_val) );
		}
	}
}
