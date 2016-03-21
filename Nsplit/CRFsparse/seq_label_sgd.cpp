#include <fstream>
#include <iostream>
#include "util.h"
#include "seq_label_sgd.h"
#include <algorithm>
#include "HashTable.h"

using namespace std;

//Function specific to Sequence Labeling Problem
SeqLabelProblem_SGD::SeqLabelProblem_SGD(char* data_file){
	
	numLabel = 0;
	N = 0;
	raw_d = -1;

	readProblem(data_file);
}

double SeqLabelProblem_SGD::inferMarg(int data_index){
	
	///All of following do only for the data sample "i"
	int i = data_index;
	
	//compute unigram factor values for i-th instance
	Seq* seq = &(data[i]);
	SeqFactor* seqFactor = &(factors[i]);
	for(int t=0;t<seq->T;t++){
		double* f_xy = seqFactor->factor_xy[t];
		Feature* fea = seq->features[t];
		for(int j=0;j<numLabel;j++){
			f_xy[j] = 0.0;
			Feature::iterator it;
			for(it=fea->begin();it!=fea->end();it++){
				f_xy[j] += w[it->first*numLabel + j] * it->second;
			}
		}
	}
	//bigram factor values
	for(int j=0;j<numLabel*numLabel;j++){
		factor_yy[j] = w[bi_offset_w+j];
	}
	//do inference on each sequence
	double** f_xy;
	double** fw;
	double** fw_inc;
	double** bk;
	
	vector<pair<int,double> >* M = new vector<pair<int,double> >[numLabel];
	double logf_yy_max = maximum(factor_yy,numLabel*numLabel);	
	double one_over_max = exp(-logf_yy_max);
	for(int i=0;i<numLabel;i++){
		int i_offset = i*numLabel;
		double value;
		for(int j=0;j<numLabel;j++){
			value = factor_yy[i_offset + j];
			if( fabs(value) > 1e-12 ){
				M[i].push_back(make_pair(j,exp(value-logf_yy_max)-one_over_max));
				//M[i].push_back(make_pair(j,exp(value)-1.0));
			}
		}
	}
	
	
	int T;
	double fw_sum;
	double logZ = 0.0;
	double logf_xy_max;
		
	T = data[i].T;
	fw = seqMargs[i].forward_msg;
	fw_inc = seqMargs[i].forward_inc_msg;
	bk = seqMargs[i].backward_msg;
	f_xy = factors[i].factor_xy;
	//forward pass
	for(int j=0;j<numLabel;j++)
		fw[0][j] = 1.0;
	normalize(fw[0],numLabel);

	for(int t=1;t<T;t++){

		logf_xy_max = maximum(f_xy[t-1],numLabel);
		for(int k=0;k<numLabel;k++)
			fw_inc[t-1][k] = fw[t-1][k] * exp(f_xy[t-1][k]-logf_xy_max);

		fw_sum = 0.0;
		for(int k=0;k<numLabel;k++){
			fw_sum += fw_inc[t-1][k];
		}
		fw_sum = fw_sum * one_over_max ;

		for(int j=0;j<numLabel;j++)
			fw[t][j] = fw_sum;

		for(int j=0;j<numLabel;j++){
			vector<pair<int,double> >::iterator it;
			for(it=M[j].begin(); it!=M[j].end(); it++){

				fw[t][it->first] += fw_inc[t-1][j]*it->second;
			}
		}

		normalize(fw[t],numLabel);
		normalize(fw_inc[t-1],numLabel);
	}
	//backward pass

	logf_xy_max = maximum(f_xy[T-1],numLabel);
	for(int j=0;j<numLabel;j++)
		bk[T-1][j] = exp( f_xy[T-1][j] - logf_xy_max );

	logZ += normalize( bk[T-1], numLabel ) + logf_xy_max;

	double bk_sum;
	for(int t=T-2;t>=0;t--){

		bk_sum = 0.0;
		for(int k=0;k<numLabel;k++){
			bk_sum += bk[t+1][k];
		}
		bk_sum = bk_sum * one_over_max;

		logf_xy_max = maximum(f_xy[t],numLabel);
		for(int j=0;j<numLabel;j++){
			double msg = bk_sum;
			vector<pair<int,double> >::iterator it;
			for(it=M[j].begin();it!=M[j].end();it++){
				msg += bk[t+1][it->first]*it->second;
			}

			bk[t][j] = msg * exp(f_xy[t][j]-logf_xy_max);
		}

		logZ += normalize( bk[t] , numLabel ) + logf_yy_max + logf_xy_max;
	}
	
	
	//get normalized marginals
	double** marg;

	T = data[i].T;
	fw = seqMargs[i].forward_msg;
	bk = seqMargs[i].backward_msg;
	marg = seqMargs[i].marginal;

	for(int t=0;t<T;t++){
		for(int j=0;j<numLabel;j++){
			marg[t][j] = fw[t][j]*bk[t][j];
		}
		normalize(marg[t],numLabel);
	}
	
	delete[] M;

	return logZ;
}

void SeqLabelProblem_SGD::applyGrad( int data_index, //input
		                        double eta, SGD* sgd  ){ //output
	int n = data_index;
	inferMarg(n);
	
	Seq* seq = &(data[n]);
    //sgd->adjust_u( 1.0/seq->T );
	SeqMarg* seqMarg = &(seqMargs[n]);
	
	//Unigram Feature
	Feature* fea;
	int w_index, w_label;
	int factor_index, fvalue_index, t_label;
	double p, loss_deriv;
	int i, fea_exp_index;
		
	double* marg;
	for(int t = 0; t<seq->T ; t++){
			
		marg = seqMarg->marginal[t];
		for(int j=0;j<numLabel;j++){
			
			t_label = seq->labels[t];
			//loss derivative
			if( j == t_label ){
				loss_deriv = marg[j] - 1.0;
			}else{
				loss_deriv = marg[j];
			}
			
			fea = seq->features[t];
			//gradient
			Feature::iterator it;
            int w_index;
			for(it=fea->begin();it!=fea->end();it++){
				w_index = it->first*numLabel + j;
                w[ w_index ] -= eta* ( loss_deriv * it->second ) ;
                sgd->penalty( w_index, w );
            }
		}
	}
    

	//Bigram Feature
	
	//Compute Mplus (M+1)
	double* Mplus = new double[numLabel*numLabel];
	double logf_yy_max = maximum(factor_yy,numLabel*numLabel);
	for(int i=0;i<numLabel*numLabel;i++){
		Mplus[i] = exp(factor_yy[i]-logf_yy_max);
	}
	double Z_yy = 0.0;
	for(int i=0;i<numLabel*numLabel;i++){
		Z_yy += Mplus[i] ;
	}
	for(int i=0;i<numLabel*numLabel;i++){
		Mplus[i] /= Z_yy;
	}
	
	
	double** fw_inc;
	double** bk;
		
	fw_inc = seqMargs[n].forward_inc_msg;
	bk = seqMargs[n].backward_msg;
	
	for(int t=0;t<data[n].T-1;t++){

		//compute normalization constant of marginal joint P(y_{t},y_{t+1})
		double Z = 0.0;
		for(int i=0;i<numLabel;i++){
			for(int j=0;j<numLabel;j++)
				Z += fw_inc[t][i]*bk[t+1][j]*Mplus[i*numLabel+j];
		}
		
		//compuate grad of bigrams
        int w_index;
		for(int i=0;i<numLabel;i++){
			for(int j=0;j<numLabel;j++){

				p = (fw_inc[t][i]*bk[t+1][j])*(Mplus[i*numLabel+j]) / Z;
				w_index = bi_offset_w+i*numLabel+j;
				if( data[n].labels[t]==i && data[n].labels[t+1]==j ){
					w[ w_index ] -= eta*( p - 1.0 );
				}else{
					w[ w_index ] -= eta * p ;
				}
                sgd->penalty( w_index, w );
			}
		}
	}

    //sgd->adjust_u((double)(seq->T));
	
	delete[] Mplus;
	//delete[] M;
}

double SeqLabelProblem_SGD::fun(){
	
	logZ_all = 0.0;
	for(int n=0;n<N;n++){
		logZ_all += inferMarg(n);
	}
	
	double** f_xy;
	double* f_yy = factor_yy;
	double log_pot_i;
	double nll = 0.0;
	for(int i=0;i<N;i++){
		
		//log potential of i-th instance
		log_pot_i = 0.0;
		f_xy = factors[i].factor_xy;
		Seq* seq = &(data[i]);
		
		log_pot_i = f_xy[0][seq->labels[0]];
		for(int t=1; t<seq->T; t++){
			log_pot_i += (f_yy[ seq->labels[t-1]*numLabel + seq->labels[t] ] + f_xy[ t ][ seq->labels[t] ]);
		}
		
		nll +=  -log_pot_i;
	}
	nll += logZ_all;
	//cerr << logZ_all << " - " << log_pot_i << endl;
	
	return nll;
}

void SeqLabelProblem_SGD::readProblem(char* data_file){
	
	/*Read Data into "seqs"
	 */
	ifstream fin(data_file);
	
	char _line[MAX_LINE];
	vector<string> tokens;
	vector<string> iv_pair;
	map<string,int>::iterator it;
	
	data.push_back(Seq());
	Seq* seq = &(data.back());
	int count=0;
    int T_total=0;
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
		if( (it=label_index_map.find(tokens[0])) == label_index_map.end() ){
			it = label_index_map.insert( make_pair(tokens[0],numLabel)).first;
			label_name_map.insert( make_pair(numLabel,tokens[0]));
			numLabel++;
		}
		labels.push_back( it->second );
		
		//Get Features
		Feature* fea = new Feature();
		for(int j=1;j<tokens.size();j++){
			split(tokens[j],":",iv_pair);
			int index = atoi(iv_pair[0].c_str());
			double value =  atof(iv_pair[1].c_str());
			fea->push_back( make_pair( index, value ) );

			if(index > raw_d)
				raw_d = index;
		}
		
		seq->features.push_back(fea);
		//next frame
		seq->T++;
		T_total++;
	}
	fin.close();
	N = data.size();
	
	int k=0;
	for(int i=0;i<N;i++){
		seq = &(data[i]);
		seq->labels = &(labels[k]);
		k += seq->T;
	}
	
	/*Read Feature Template
	 */
	if( param.info_file == NULL ){
		cerr << "exit: Sequence Label Problem needs feature template specified as info file" << endl;
		exit(0);
	}
	
	ifstream finfo(param.info_file);
	
	//filter one line
	finfo.getline(_line,MAX_LINE);
	//get unigram template
	finfo.getline(_line,MAX_LINE);
	string line(_line);
	split(line," ",tokens);
	
	for(int i=0;i<tokens.size();i++){
		uni_fea_template.push_back(atoi(tokens[i].c_str()));
	}
	finfo.close();
	
	d_expand = compute_d_expand( raw_d );
	
	/*Expand feature for each data sample
	 */
	Feature fea_exp;
	int fea_index;
	for(int i=0;i<N;i++){
		Seq* seq = &(data[i]);
		for(int t=0;t<seq->T;t++){
			
			feature_expand( *seq, t, fea_exp );
			*(seq->features[t]) = fea_exp;
		}
	}
	
	
	/*Build Model Parameter Array
	 */
	bi_offset_w = (1+raw_d) * uni_fea_template.size() * numLabel;
	d =  bi_offset_w  +   numLabel*numLabel;
	
	w = new double[d];
	for(int i=0;i<d;i++)
		w[i] = 0.0;
	
	/*Build Factor Table Array
	 */
	n = 0;
	for(int i=0;i<N;i++){
		n += data[i].T * numLabel; //unigram factor
	}
	bi_offset_fv = n;
	n += numLabel * numLabel ; //bigram factor
		
	fvalue = new double[n];
	for(int i=0;i<n;i++)
		fvalue[i] = 0.0;
	
	k=0;
	factors = new SeqFactor[N];
	for(int i=0;i<N;i++){
		factors[i].factor_xy = new double*[data[i].T];
		for(int t=0;t<data[i].T;t++){

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
	for(int i=0;i<T_total;i++){
		forward_msg[i] = new double[numLabel];
		forward_inc_msg[i] = new double[numLabel];
		backward_msg[i] = new double[numLabel];
		marginal[i] = new double[numLabel];
	}
	
	k = 0;
	seqMargs = new SeqMarg[N];
	for(int i=0;i<N;i++){
		
		seqMargs[i].forward_msg = &(forward_msg[k]);
		seqMargs[i].forward_inc_msg = &(forward_inc_msg[k]);
		seqMargs[i].backward_msg = &(backward_msg[k]);
		seqMargs[i].marginal = &(marginal[k]);
		
		k += data[i].T;
	}
}

int SeqLabelProblem_SGD::compute_d_expand(int raw_d){
	
	return 1 + raw_d * uni_fea_template.size();
}

void SeqLabelProblem_SGD::feature_expand(Seq& seq, int t, Feature& fea_exp){
	
	fea_exp.clear();
	
	pair<int,double>* iv_pair;
	int index,t2;
	for(int i=0;i<uni_fea_template.size();i++){
		t2 = t + uni_fea_template[i];
		if( t2 < 0 || t2 >= seq.T )
			continue;
		Feature* feature = seq.features[t2];
		for(int j=0;j<feature->size();j++){
			iv_pair = &(feature->at(j));
			index = i*(1+raw_d) + iv_pair->first;
			fea_exp.push_back( make_pair( index, iv_pair->second ) );
		}
	}
}
