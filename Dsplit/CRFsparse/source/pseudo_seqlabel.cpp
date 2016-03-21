#include <fstream>
#include <iostream>
#include <algorithm>
#include "util.h"
#include "pseudo_seqlabel.h"
#include "seq_label.h"

using namespace std;


//Functions for PseudoSeqLabelProblem only
PseudoSeqLabelProblem::PseudoSeqLabelProblem(char* data_file){

	K = 0;
	N = 0;
	d = raw_d = -1;
	readProblem(data_file);
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

PseudoSeqLabelProblem::PseudoSeqLabelProblem(char* model_file, char* data_file){

	K = 0;
	N = 0;
	d = raw_d = -1;
	readProblem(model_file, data_file);
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

//Functions for "Problem" Interface
void PseudoSeqLabelProblem::compute_fv_change(double* w_change, vector<Int>& act_set, //input
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
	//unigram feature
	for(i=0; i<act_set.size() && act_set[i] < w_bi_offset ;i++){

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
	//bigram feature
	Int left_fea_index, right_fea_index, left_w_label, right_w_label;
	for(; i<act_set.size(); i++){
		w_index = act_set[i];
		delta_w = w_change[ w_index ];
		//find factors related to w_index
		right_w_label = left_fea_index = (w_index-w_bi_offset) / K;
		left_w_label = right_fea_index = (w_index-w_bi_offset) % K;
		
		feature = &(bi_left_inv[ left_fea_index ]);
		for(it=feature->begin();it!=feature->end();it++){
			fv_index = it->first + left_w_label;
			fv_change_table[fv_index] += delta_w * it->second;
		}
		feature = &(bi_right_inv[ right_fea_index ]);
		for(it=feature->begin();it!=feature->end();it++){
			fv_index = it->first + right_w_label;
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

void PseudoSeqLabelProblem::update_fvalue(vector<pair<Int,double> >& fv_change, double scalar){
	
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

void PseudoSeqLabelProblem::inferMarg(){

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

double PseudoSeqLabelProblem::collect_grad( Int w_label, Feature* fea ){
	
	double g = 0.0;
	Feature::iterator it;
	for(it=fea->begin();it!=fea->end();it++){

		Int sample_index = (it->first/K);
		Int true_label = labels[sample_index];
		
		//loss derivative
		double loss_deriv;
		if( w_label == true_label ){
			loss_deriv = marginal[sample_index][w_label] - 1.0;
			//cerr << label_name_map.find(w_label)->second << ":" << marginal[factor_index][w_label]-1.0 << endl;
		}else
			loss_deriv = marginal[sample_index][w_label];
		
		//gradient
		g += loss_deriv * it->second;
	}
	
	return g;
}

void PseudoSeqLabelProblem::grad( vector<Int>& act_set, //input
		vector<double>& grad){ //output

	grad.clear();
	
	//for each active feature, collect residual at each factor (sample)
	Feature* fea;
	Int w_index, w_label;
	Int sample_index, fvalue_index;
	double g;
	Int ind, fea_index;
	//unigram features
	for(ind=0; ind<act_set.size() && act_set[ind] < w_bi_offset ; ind++){
		
		w_index = act_set[ind];
		fea_index = w_index / K ;
		w_label = w_index % K;
		
		fea = &(data_inv[fea_index]);
		g = collect_grad(w_label, fea);
		
		g += param.theta*w[w_index]; //L2-regularization
		grad.push_back(g);
	}
	//bigram features
	Int left_fea_index, right_fea_index, left_w_label, right_w_label;
	for(; ind<act_set.size(); ind++){
		w_index = act_set[ind];
		right_w_label = left_fea_index = (w_index-w_bi_offset) / K;
		left_w_label = right_fea_index = (w_index-w_bi_offset) % K;
		//collect residual from left bigram samples
		fea = &(bi_left_inv[left_fea_index]);
		g = collect_grad( left_w_label, fea );
		fea = &(bi_right_inv[right_fea_index]);
		g += collect_grad( right_w_label, fea);
		
		g += param.theta2*w[w_index];
		grad.push_back(g);
	}
}

double PseudoSeqLabelProblem::fun(){

	double** f_xy;
	double nll = 0.0;
	for(Int i=0;i<N;i++){
		
		//log potential of i-th instance
		nll += -factor_xy[i][ labels[i] ];
	}
	nll += logZ;
	//cerr << logZ << " - " << log_pot_i << endl;
	
	double w_uni_norm_sq=0.0;
	for(Int j=0;j<w_bi_offset;j++)
		w_uni_norm_sq += w[j]*w[j];
	
	double w_bi_norm_sq=0.0;
	for(Int j=w_bi_offset; j<d; j++){
		w_bi_norm_sq += w[j]*w[j];
	}
	
	return nll + param.theta*w_uni_norm_sq/2.0 + param.theta2*w_bi_norm_sq/2.0;
}

/** Given a w, compute all factor values.
 */
void PseudoSeqLabelProblem::compute_fvalue(){
	
	sample_updated.clear();
	for(Int i=0;i<n;i++){
		fvalue[i] = 0.0; 
	}
	//unigram factor
	Int findex, windex;
	for(Int i=0;i<N;i++){
		sample_updated.insert(i);
		for(Int k=0;k<K;k++){

			findex = i*K + k; 
			//unigram feature
			Feature* fea = data[i];
			for(Feature::iterator it=fea->begin();it!=fea->end();it++){
				windex = it->first * K + k;
				fvalue[findex] += w[ windex ]*it->second;
			}
			//left bigram feature
			fea = data_bi_left[i];
			for(Feature::iterator it=fea->begin();it!=fea->end();it++){
				windex = w_bi_offset + it->first * K + k;
				fvalue[findex] += w[ windex ]*it->second;
			}
			//right bigram feature
			fea = data_bi_right[i];
			for(Feature::iterator it=fea->begin();it!=fea->end();it++){
				windex = w_bi_offset + k * K + it->first;
				fvalue[findex] += w[ windex ]*it->second;
			}
		}
	}
	
	inferMarg();
}

double PseudoSeqLabelProblem::train_accuracy(){
	double n_hit = 0.0;
	Int pred;
	for(Int i=0;i<N;i++){

		maximum(marginal[i], K, pred);
		n_hit += (pred == labels[i]) ;
	}

	return n_hit / N ;
}


void PseudoSeqLabelProblem::test_accuracy(const char* output_file){

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


void PseudoSeqLabelProblem::readProblem(char* model_file, char* data_file){

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

void PseudoSeqLabelProblem::readProblem(char* data_file){

	char* _line = new char[MAX_LINE];
	string line;
	vector<string> tokens;
	vector<string> iv_pair;

	/*Read Feature Template
	*/
	if( param.info_file == NULL ){
		cerr << "exit: PseudoSeqLabel Problem needs feature template specified as info file" << endl;
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
	
	/* Read Sequences from data file
	 */
	vector<Seq> seq_data;
	ifstream fin(data_file);
	if (fin.fail()){
		cerr<< "can't open data file."<<endl;
		exit(0);
	}
	seq_data.push_back(Seq());
	Seq* seq = &(seq_data.back());
	Int count_line = 1;
	Int max_index = -1;
	while( !fin.eof() ){
		fin.getline(_line,MAX_LINE);

		string line(_line);
		if( line.size() <= 1 && !fin.eof() ){
			seq_data.push_back(Seq());
			seq = &(seq_data.back());
			continue;
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

		seq->features.push_back(fea);
		//next frame
		seq->T++;

		count_line++;
	}
	fin.close();
	if(seq_data.back().T == 0 ){
		seq_data.pop_back();
	}
	Int t_cum=0;
	for(Int i=0;i<seq_data.size();i++){
		Seq* seq = &(seq_data[i]);
		seq->labels = &(labels[t_cum]);
		t_cum += seq->T;
	}
	cerr << "|seq|=" << seq_data.size() << endl;
	/** Convert Seq Data to Data in the pseudolikelihood principle
	 */
	for(Int i=0;i<seq_data.size();i++){
		Seq* seq = &(seq_data[i]);
		
		for(Int t=0;t<seq->T;t++){
			Feature* fea = seq->features[t];
			data.push_back(fea);
			
			Feature* bi_fea_left = new Feature();
			Feature* bi_fea_right = new Feature();
			
			if( t-1 >= 0 )
				bi_fea_left->push_back( make_pair(seq->labels[t-1],1.0) );
			if( t+1 < seq->T )
				bi_fea_right->push_back( make_pair(seq->labels[t+1],1.0) );
			
			data_bi_left.push_back(bi_fea_left);
			data_bi_right.push_back(bi_fea_right);
		}
	}
	N = data.size();
	cerr << "N=" << N << endl;
	if( raw_d == -1 ){ 
		raw_d = max_index;
	}
	w_bi_offset = (1+raw_d)*K;
	d = w_bi_offset + K*K;
	cerr << "d=" << d << endl;
	w = new double[d];
	for(Int i=0;i<d;i++)
		w[i] = 0.0;
	
	/*Build data inverted index
	*/
	build_data_inv();
	
	/*Build Factor Table Array
	*/
	n = N*K;
	cerr << "n=" << n << endl;
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

void PseudoSeqLabelProblem::build_data_inv(){

	//initialize data_inv
	for(Int i=0;i<(w_bi_offset/K);i++)
		data_inv.push_back(Feature());
	for(Int i=0;i<K;i++){
		bi_left_inv.push_back(Feature());
		bi_right_inv.push_back(Feature());
	}

	//scan data, expand feature, add to data_inv
	Feature fea;
	Int fea_index;
	double fea_val;
	for(Int i=0;i<N;i++){
		Feature* fea = data[i];
		for(Feature::iterator it=fea->begin(); it!=fea->end(); it++){
			data_inv[it->first].push_back( make_pair(i*K, it->second) );
		}
	}

	//scan bigram left (right) data, add to bi_left_inv (bi_right_inv)
	for(Int i=0;i<N;i++){
		Feature* fea = data_bi_left[i];
		for(Feature::iterator it=fea->begin(); it!=fea->end(); it++){
			bi_left_inv[it->first].push_back( make_pair(i*K, it->second) );
		}
	}
	for(Int i=0;i<N;i++){
		Feature* fea = data_bi_right[i];
		for(Feature::iterator it=fea->begin(); it!=fea->end(); it++){
			bi_right_inv[it->first].push_back( make_pair(i*K, it->second) );
		}
	}
}
