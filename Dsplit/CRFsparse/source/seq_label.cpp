#include <fstream>
#include <iostream>
#include "util.h"
#include "seq_label.h"
#include <algorithm>

using namespace std;

//Function specific to Sequence Labeling Problem
SeqLabelProblem::SeqLabelProblem(char* data_file){

	numLabel = 0;
	N = 0;
	raw_d = -1;
	readProblem(data_file);
	bi_factor_updated = true;
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

SeqLabelProblem::SeqLabelProblem(char* model_file, char* data_file){

	numLabel = 0;
	N = 0;
	raw_d = -1;
	readProblem(model_file, data_file);
	bi_factor_updated = true;
	for(Int i=0;i<N;i++)
		sample_updated.insert(i);
	inferMarg();
}

//Functions for "Problem"
void SeqLabelProblem::compute_fv_change(double* w_change, vector<Int>& act_set, //input
		vector<pair<Int,double> >& fv_change){ //output

	double delta_w;
	Int w_index;
	Feature* feature;
	Feature::iterator it;

	//unigram features
	double* fv_change_table = new double[n];
	for(Int i=0;i<n;i++)
		fv_change_table[i] = 0.0;

	Int i;
	Int fv_index;
	Int label;
	pair<Int,double>* pair;
	for(i=0; i<act_set.size() && act_set[i]<bi_offset_w ;i++){

		w_index = act_set[i];
		delta_w = w_change[ w_index ];
		label = w_index % numLabel;
		//find factors(w_index)
		feature = &(data_inv[ w_index/numLabel ]);
		for(it=feature->begin();it!=feature->end();it++){

			fv_index = it->first + label;
			fv_change_table[fv_index] += delta_w * it->second;
		}
	}

	//bigram features
	for(;i<act_set.size();i++){

		w_index = act_set[i];
		delta_w = w_change[ w_index ];

		fv_index = bi_offset_fv + (w_index - bi_offset_w);
		fv_change_table[fv_index] = delta_w;
	}

	//copy to vector
	fv_change.clear();
	for(Int i=0;i<n;i++)
		if(fv_change_table[i] != 0.0){
			fv_change.push_back(make_pair(i,fv_change_table[i]));
		}

	delete[] fv_change_table;
}

void SeqLabelProblem::update_fvalue(vector<pair<Int,double> >& fv_change, double scalar){

	sample_updated.clear();

	vector<Int> factor_updated;
	factor_updated.resize(fv_change.size());

	Int prev_factor = -1, factor;
	vector<pair<Int,double> >::iterator it;
	for(it=fv_change.begin();it!=fv_change.end(); it++){

		fvalue[it->first] += scalar * it->second;

		//record which factors have been updated
		if( it->first < bi_offset_fv ){
			factor = it->first/numLabel;
			if( factor != prev_factor ){
				factor_updated.push_back(factor);
			}
		}else{
			bi_factor_updated = true;
		}
		prev_factor = factor;
	}

	//record which samples have been updated
	Int factor_index = 0;
	Int j=0, prev_j=0;
	for(Int i=0;i<N;i++){
		factor_index += data[i].T;
		for( ;j<factor_updated.size() && factor_updated[j]<factor_index; j++);

		if( j != prev_j ){
			sample_updated.insert(i);
			prev_j = j;
		}

		if( j>=factor_updated.size() )
			break;
	}
	inferMarg();
}

void SeqLabelProblem::inferMax(){
	
	//construct a sparse transition matrix
	vector<pair<Int,double> >* fyy_tr_sparse = new vector<pair<Int,double> >[numLabel];
	set<Int>* fyy_tr_nonzeros = new set<Int>[numLabel];
	for(Int i=0;i<numLabel;i++){
		Int i_offset = i*numLabel;
		double value;
		for(Int j=0;j<numLabel;j++){
			value = factor_yy[i_offset + j];
			if( fabs(value) > 1e-5 ){
				fyy_tr_sparse[j].push_back(make_pair(i,value));
				fyy_tr_nonzeros[j].insert(i);
			}
		}
	}
	
	//compute Viterbi
	seqMaxs = new SeqMax[N];
	vector<Int> index;
	for(Int j=0;j<numLabel;j++)
		index.push_back(j);
	
	for (Int i=0;i<N;i++){
		
		Int T = data[i].T;
		seqMaxs[i].alpha = new double*[T];
		seqMaxs[i].traceBack = new Int*[T-1];
		double** alpha = seqMaxs[i].alpha; 
		Int** tB = seqMaxs[i].traceBack; 
		for (Int j=0;j<T;j++){
			alpha[j] = new double[numLabel];
			if(j<T-1)
				tB[j] = new Int[numLabel];
		}
		
		double** f_xy = factors[i].factor_xy;
		
		for (Int j=0;j<numLabel;j++){
			alpha[0][j] = f_xy[0][j];
		}
		for (Int t=1;t<T;t++){
			sort( index.begin(), index.end(), ScoreComp(alpha[t-1]) );
			for (Int k=0;k<numLabel;k++){
				vector<pair<Int,double> >::iterator it;
				//search for f_yy != 0
				double val, max=-1e300;
				Int argmax;
				for(it=fyy_tr_sparse[k].begin(); it!=fyy_tr_sparse[k].end(); it++){
					val = alpha[t-1][it->first] + it->second;
					if( val > max ){
						max = val;
						argmax = it->first;
					}
				}
				//search for f_yy == 0.0
				Int r=0;
				while( r<index.size() && 
						fyy_tr_nonzeros[k].find(index[r])!=fyy_tr_nonzeros[k].end() ) r++;
				
				if( r < index.size() ){
					Int j = index[r];
					if( alpha[t-1][j] > max ){
						max = alpha[t-1][j];
						argmax = j;
					}
				}
				
				//take argmax 
				alpha[t][k] = f_xy[t][k] + max;
				tB[t-1][k] = argmax;
			}
		}
		
		seqMaxs[i].maxLabel = new Int[T];
		Int* mL = seqMaxs[i].maxLabel;
		maximum(alpha[T-1],numLabel,mL[T-1]);
		for (Int t=T-2;t>=0;t--){
			mL[t] = tB[t][ mL[t+1] ];
		}
		//delete alpha, traceback
		for(Int j=0;j<T;j++){
			delete[] seqMaxs[i].alpha[j];
			if(j<T-1) delete[] seqMaxs[i].traceBack[j];
		}
		delete[] seqMaxs[i].alpha;
		delete[] seqMaxs[i].traceBack;
	}
	
	delete[] fyy_tr_sparse;
	delete[] fyy_tr_nonzeros;
}

void SeqLabelProblem::inferMarg(){

	//do inference on each sequence
	double** f_xy;
	double** fw;
	double** fw_inc;
	double** bk;

	vector<pair<Int,double> >* M = new vector<pair<Int,double> >[numLabel];
	double logf_yy_max = maximum(factor_yy,numLabel*numLabel);	
	double one_over_max = exp(-logf_yy_max);
	for(Int i=0;i<numLabel;i++){
		Int i_offset = i*numLabel;
		double value;
		for(Int j=0;j<numLabel;j++){
			value = factor_yy[i_offset + j];
			if( fabs(value) > 1e-5 ){
				M[i].push_back(make_pair(j,exp(value-logf_yy_max)-one_over_max));
				//M[i].push_back(make_pair(j,exp(value)-1.0));
			}
		}
	}
	Int T;
	double fw_sum;
	logZ = 0.0;
	double logf_xy_max;
	for(Int i=0;i<N;i++){

		if( !bi_factor_updated && sample_updated.find(i) == sample_updated.end() ){
			continue;
		}

		T = data[i].T;
		fw = seqMargs[i].forward_msg;
		fw_inc = seqMargs[i].forward_inc_msg;
		bk = seqMargs[i].backward_msg;
		f_xy = factors[i].factor_xy;

		//forward pass
		for(Int j=0;j<numLabel;j++)
			fw[0][j] = 1.0;
		normalize(fw[0],numLabel);

		for(Int t=1;t<T;t++){

			logf_xy_max = maximum(f_xy[t-1],numLabel);
			for(Int k=0;k<numLabel;k++)
				fw_inc[t-1][k] = fw[t-1][k] * exp(f_xy[t-1][k]-logf_xy_max);

			fw_sum = 0.0;
			for(Int k=0;k<numLabel;k++){
				fw_sum += fw_inc[t-1][k];
			}
			fw_sum = fw_sum * one_over_max ;

			for(Int j=0;j<numLabel;j++)
				fw[t][j] = fw_sum;

			for(Int j=0;j<numLabel;j++){
				vector<pair<Int,double> >::iterator it;
				for(it=M[j].begin(); it!=M[j].end(); it++){

					fw[t][it->first] += fw_inc[t-1][j]*it->second;
				}
			}

			normalize(fw[t],numLabel);
			normalize(fw_inc[t-1],numLabel);
		}
		//backward pass

		logf_xy_max = maximum(f_xy[T-1],numLabel);
		for(Int j=0;j<numLabel;j++)
			bk[T-1][j] = exp( f_xy[T-1][j] - logf_xy_max );

		logZ += normalize( bk[T-1], numLabel ) + logf_xy_max;

		double bk_sum;
		for(Int t=T-2;t>=0;t--){

			bk_sum = 0.0;
			for(Int k=0;k<numLabel;k++){
				bk_sum += bk[t+1][k];
			}
			bk_sum = bk_sum * one_over_max;

			logf_xy_max = maximum(f_xy[t],numLabel);
			for(Int j=0;j<numLabel;j++){
				double msg = bk_sum;
				vector<pair<Int,double> >::iterator it;
				for(it=M[j].begin();it!=M[j].end();it++){
					msg += bk[t+1][it->first]*it->second;
				}

				bk[t][j] = msg * exp(f_xy[t][j]-logf_xy_max);
			}

			logZ += normalize( bk[t] , numLabel ) + logf_yy_max + logf_xy_max;
		}
	}


	//get normalized marginals
	double** marg;
	for(Int i=0;i<N;i++){

		if( !bi_factor_updated && sample_updated.find(i) == sample_updated.end() ){
			continue;
		}

		T = data[i].T;
		fw = seqMargs[i].forward_msg;
		bk = seqMargs[i].backward_msg;
		marg = seqMargs[i].marginal;

		for(Int t=0;t<T;t++){
			for(Int j=0;j<numLabel;j++){
				marg[t][j] = fw[t][j]*bk[t][j];
			}
			normalize(marg[t],numLabel);
		}
	}

	delete[] M;
}

void SeqLabelProblem::grad( vector<Int>& act_set, //input
		vector<double>& grad){ //output

	grad.clear();

	//Unigram Feature
	Feature* feature;
	Int w_index, w_label;
	Int factor_index, fvalue_index, t_label;
	double g, loss_deriv;
	Int uni_ind, fea_exp_index;
	for(uni_ind=0; uni_ind<act_set.size() && act_set[uni_ind]<bi_offset_w ;uni_ind++){

		w_index = act_set[uni_ind];
		fea_exp_index = w_index / numLabel ;
		w_label = w_index % numLabel;

		feature = &(data_inv[fea_exp_index]);

		g = 0.0;
		vector<pair<Int,double> >::iterator it;
		for(it=feature->begin();it!=feature->end();it++){

			factor_index = (it->first/numLabel);
			t_label = labels[factor_index];

			//loss derivative
			if( w_label == t_label ){
				loss_deriv = marginal[factor_index][w_label] - 1.0;
				//cerr << label_name_map.find(w_label)->second << ":" << marginal[factor_index][w_label]-1.0 << endl;
			}else
				loss_deriv = marginal[factor_index][w_label];

			//gradient
			g += loss_deriv * it->second;
		}

		g += param.theta*w[w_index]; //L2-regularization
		grad.push_back(g);
	}

	//Bigram Feature
	Int nnz_yy = act_set.size()-uni_ind;
	for(Int j=0;j<nnz_yy;j++)
		grad.push_back(0.0);

	//Compute Mplus (M+1)
	double* Mplus = new double[numLabel*numLabel];
	double logf_yy_max = maximum(factor_yy,numLabel*numLabel);
	for(Int i=0;i<numLabel*numLabel;i++){
		Mplus[i] = exp(factor_yy[i]-logf_yy_max);
	}
	double Z_yy = 0.0;
	for(Int i=0;i<numLabel*numLabel;i++){
		Z_yy += Mplus[i] ;
	}
	for(Int i=0;i<numLabel*numLabel;i++){
		Mplus[i] /= Z_yy;
	}

	//Compute M
	/*vector<pair<Int,double> >* M  = new vector<pair<Int,double> >[numLabel];
	  double one_over_max = exp(-logf_yy_max)/Z_yy;
	  for(Int i=0;i<numLabel;i++){
	  for(Int j=0;j<numLabel;j++){
	  if( fabs(Mplus[i]-one_over_max) > 1e-12 )
	  M[i].push_back(make_pair(j,Mplus[i*numLabel+j]-one_over_max));
	  }
	  }*/

	double** fw_inc;
	double** bk;
	for(Int n=0;n<N;n++){

		fw_inc = seqMargs[n].forward_inc_msg;
		bk = seqMargs[n].backward_msg;

		for(Int t=0;t<data[n].T-1;t++){

			//compute normalization constant of marginal joInt P(y_{t},y_{t+1})
			/*double Z = one_over_max; //(fw_inc_sum)(bk_sum) = 1
			  for(Int i=0;i<numLabel;i++){
			  vector<pair<Int,double> >::iterator it;
			  for(it=M[i].begin();it!=M[i].end();it++){
			  Z += it->second*fw_inc[t][i]*bk[t+1][it->first];
			  }
			}*/
			double Z = 0.0;
			for(Int i=0;i<numLabel;i++){
				for(Int j=0;j<numLabel;j++)
					Z += fw_inc[t][i]*bk[t+1][j]*Mplus[i*numLabel+j];
			}

			//compuate grad of bigrams
			Int lab1,lab2,w_index, w_bi_index;
			double p;
			for(Int j=0;j<nnz_yy;j++){
				w_index = act_set[uni_ind+j];
				w_bi_index = w_index-bi_offset_w;
				lab1 = w_bi_index / numLabel;
				lab2 = w_bi_index % numLabel;

				p = (fw_inc[t][lab1]*bk[t+1][lab2])*(Mplus[w_bi_index]) / Z;

				if( data[n].labels[t]==lab1 && data[n].labels[t+1]==lab2 ){
					grad[uni_ind+j] += p - 1.0;
				}else{
					grad[uni_ind+j] += p;
				}
				
				grad[uni_ind+j] += param.theta2*w[w_index];
			}
		}
	}

	delete[] Mplus;
}

double SeqLabelProblem::fun(){

	double** f_xy;
	double* f_yy = factor_yy;
	double log_pot_i;
	double nll = 0.0;
	for(Int i=0;i<N;i++){

		//log potential of i-th instance
		log_pot_i = 0.0;
		f_xy = factors[i].factor_xy;
		Seq* seq = &(data[i]);

		log_pot_i = f_xy[0][seq->labels[0]];
		for(Int t=1; t<seq->T; t++){
			log_pot_i += (f_yy[ seq->labels[t-1]*numLabel + seq->labels[t] ] + f_xy[ t ][ seq->labels[t] ]);
		}

		nll +=  -log_pot_i;
	}
	nll += logZ;
	
	double w_uni_norm_sq=0.0;
	for(Int j=0;j<bi_offset_w;j++)
		w_uni_norm_sq += w[j]*w[j];
	
	double w_bi_norm_sq=0.0;
	for(Int j=bi_offset_w; j<d; j++){
		w_bi_norm_sq += w[j]*w[j];
	}
	
	//cerr << logZ << " - " << log_pot_i << endl;
	
	return nll + param.theta*w_uni_norm_sq/2.0 + param.theta2*w_bi_norm_sq/2.0;
}

void SeqLabelProblem::compute_fvalue(){

	for(Int i=0;i<n;i++){
		fvalue[i] = 0.0; 
	}    
	//unigram factor
	Int findex, windex;
	Int t = 0; 
	for(Int i=0;i<N;i++){
		for(Int j=0;j<data[i].T;j++){

			Feature* fea = data[i].features[j];
			for(Int k=0;k<numLabel;k++){

				findex = t*numLabel + k; 
				Feature::iterator it;
				//cerr << i << ", " << j << ", " << k << endl;
				for(it=fea->begin();it!=fea->end();it++){
					windex = it->first * numLabel + k; 
					fvalue[findex] += w[ windex ]*it->second;
				}    
			}    
			t++; 
		}    
	}    
	//bigram factor

	for(Int i=0;i<numLabel*numLabel;i++){
		fvalue[bi_offset_fv+i] = w[bi_offset_w+i];
	}    

}

double SeqLabelProblem::train_accuracy(){
	double error = 0.0;
	Int pred;
	for(Int i=0;i<N;i++){

		Seq* seq = &(data[i]);
		SeqMarg* seqMarg = &(seqMargs[i]);

		for(Int t=0; t<seq->T; t++){
			maximum(seqMarg->marginal[t],numLabel,pred);
			error += (pred == seq->labels[t]) ;
		}
	}

	return error / T_total ;
}


void SeqLabelProblem::test_accuracy(const char* output_file){

	ofstream fout(output_file);
	if( param.predict_method > 1 ){
		double eta = (double)param.predict_method;
		cerr <<"eta=" << eta << endl;
		for(Int i=0;i<d;i++){
			w[i] = w[i]*eta;
		}
	}
	compute_fvalue();
	if( param.predict_method == 0 )
		inferMax();
	if( param.predict_method >= 1 )
		inferMarg();

	double accuracy = 0.0;
	Int pred;
	for(Int i=0;i<N;i++){

		Seq* seq = &(data[i]);
		
		for(Int t=0; t<seq->T; t++){
			if( param.predict_method==0 )
				pred = seqMaxs[i].maxLabel[t];
			else
				maximum(seqMargs[i].marginal[t],numLabel,pred);
			
			fout << label_name_map[pred]<<endl; 
			accuracy += (pred == seq->labels[t]) ;
		}
		fout<<endl;
	}
	cerr << "testing accuracy: " << accuracy / T_total <<endl;

	fout.close();
};


void SeqLabelProblem::readProblem(char* model_file, char* data_file){

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

void SeqLabelProblem::readProblem(char* data_file){

	char* _line = new char[MAX_LINE];
	string line;
	vector<string> tokens;
	vector<string> iv_pair;

	/*Read Feature Template
	*/
	if( param.info_file == NULL ){
		cerr << "exit: Sequence Label Problem needs feature template specified as info file" << endl;
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
	numLabel = label_index_map.size();
	for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++)
		label_name_map.insert(make_pair(it->second,it->first));

	//filter one line ("unigram:")
	finfo.getline(_line,MAX_LINE); 
	finfo.getline(_line,MAX_LINE); 
	line = string(_line);
	split(line," ",tokens);
	for(Int i=0;i<tokens.size();i++){
		uni_fea_template.push_back(atoi(tokens[i].c_str()));
	}
	//ignore "bigram:" part of info file, and use default bi-gram template currently
	finfo.close();

	/* Read Sequences from data file
	*/
	ifstream fin(data_file);
	if (fin.fail()){
		cerr<< "can't open data file."<<endl;
		exit(0);
	}
	data.push_back(Seq());
	Seq* seq = &(data.back());
	Int count_line = 1;
	T_total=0;

	Int max_index = -1;
	while( !fin.eof() ){
		fin.getline(_line,MAX_LINE);

		string line(_line);
		if( line.size() <= 1 && !fin.eof() ){
			data.push_back(Seq());
			seq = &(data.back());
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
		T_total++;

		count_line++;
	}
	fin.close();
	
	while (data.back().T == 0)
		data.pop_back();
	N = data.size();
	if( raw_d == -1 ) 
		raw_d = max_index;

	Int k=0;
	for(Int i=0;i<N;i++){
		seq = &(data[i]);
		seq->labels = &(labels[k]);
		k += seq->T;
	}

	/* Based on "raw_d" obtained from data file, and generate feature expension based on feature template
	*/
	d_expand = compute_d_expand( raw_d );
	/*Build Model Parameter Array
	*/
	bi_offset_w = (1+raw_d) * uni_fea_template.size() * numLabel;
	d =  bi_offset_w  +   numLabel*numLabel;

	w = new double[d];
	for(Int i=0;i<d;i++)
		w[i] = 0.0;
	/*Build data inverted index
	*/
	build_data_inv();
	/*Build Factor Table Array
	*/
	n = 0;
	for(Int i=0;i<N;i++){
		n += data[i].T * numLabel; //unigram factor
	}
	bi_offset_fv = n;
	n += numLabel * numLabel ; //bigram factor

	fvalue = new double[n];
	for(Int i=0;i<n;i++)
		fvalue[i] = 0.0;

	k=0;
	factors = new SeqFactor[N];
	for(Int i=0;i<N;i++){
		factors[i].factor_xy = new double*[data[i].T];
		for(Int t=0;t<data[i].T;t++){

			factors[i].factor_xy[t] = &(fvalue[k]);
			k += numLabel;
		}
	}
	
	//Bigram 
	factor_yy = &(fvalue[k]);
	
	/*Build Forward-Backward msg array for each seq
	*/
	forward_msg = new double*[T_total];
	forward_inc_msg = new double*[T_total];
	backward_msg = new double*[T_total];
	marginal = new double*[T_total];
	for(Int i=0;i<T_total;i++){
		forward_msg[i] = new double[numLabel];
		forward_inc_msg[i] = new double[numLabel];
		backward_msg[i] = new double[numLabel];
		marginal[i] = new double[numLabel];
	}
	k = 0;
	seqMargs = new SeqMarg[N];
	for(Int i=0;i<N;i++){

		seqMargs[i].forward_msg = &(forward_msg[k]);
		seqMargs[i].forward_inc_msg = &(forward_inc_msg[k]);
		seqMargs[i].backward_msg = &(backward_msg[k]);
		seqMargs[i].marginal = &(marginal[k]);

		k += data[i].T;
	}
	
	delete[] _line;
}

void SeqLabelProblem::build_data_inv(){

	//initialize data_inv
	for(Int i=0;i<d_expand;i++)
		data_inv.push_back(Feature());

	//scan data, expand feature, add to data_inv
	Feature fea_exp;
	Int factor_index=0;
	Int fea_index;
	for(Int i=0;i<N;i++){
		Seq* seq = &(data[i]);
		for(Int t=0;t<seq->T;t++){
			feature_expand( *seq, t, fea_exp );

			for(Int j=0;j<fea_exp.size();j++){
				fea_index = fea_exp[j].first;
				data_inv[fea_index].push_back( make_pair(factor_index*numLabel, fea_exp[j].second) );
			}
			factor_index++;
		}
	}
}

Int SeqLabelProblem::compute_d_expand(Int raw_d){

	return 1 + raw_d * uni_fea_template.size();
}

void SeqLabelProblem::feature_expand(Seq& seq, Int t, Feature& fea_exp){

	fea_exp.clear();

	pair<Int,double>* iv_pair;
	Int index,t2;
	for(Int i=0;i<uni_fea_template.size();i++){
		t2 = t + uni_fea_template[i];
		if( t2 < 0 || t2 >= seq.T )
			continue;
		Feature* feature = seq.features[t2];
		for(Int j=0;j<feature->size();j++){
			iv_pair = &(feature->at(j));
			index = i*(1+raw_d) + iv_pair->first;
			fea_exp.push_back( make_pair( index, iv_pair->second ) );
		}
	}
}
