#include <fstream>
#include <iostream>
#include "seq_label.h"
#include <algorithm>

using namespace std;


SeqLabelProblem::SeqLabelProblem(int _raw_d, int L){
	
	N = 0;
	raw_d = _raw_d;
	numLabel = L;
	
	factors = NULL;
	factor_yy =  NULL;
	seqMargs = NULL;
	forward_msg = NULL;
	backward_msg = NULL;
	forward_inc_msg = NULL;
	marginal = NULL;
}

//Functions for "Problem"
void SeqLabelProblem::compute_fv_change(double* w_change, vector<int>& act_set, //input
		vector<pair<int,double> >& fv_change){ //output

	double delta_w;
	int w_index;
	Feature* feature;
	Feature::iterator it;

	//unigram features
	double* fv_change_table = new double[n];
	for(int i=0;i<n;i++)
		fv_change_table[i] = 0.0;

	int i;
	int fv_index;
	int label;
	pair<int,double>* pair;
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
	for(int i=0;i<n;i++)
		if(fv_change_table[i] != 0.0){
			fv_change.push_back(make_pair(i,fv_change_table[i]));
		}

	delete[] fv_change_table;
}

void SeqLabelProblem::update_fvalue(vector<pair<int,double> >& fv_change, double scalar){

	sample_updated.clear();

	vector<int> factor_updated;
	factor_updated.resize(fv_change.size());

	int prev_factor = -1, factor;
	vector<pair<int,double> >::iterator it;
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
	}

	//record which samples have been updated
	int factor_index = 0;
	int j=0, prev_j=0;
	for(int i=0;i<N;i++){
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

void SeqLabelProblem::compute_fvalue(){
	
	for(int i=0;i<n;i++){
		fvalue[i] = 0.0;
	}
	//unigram factor
	int findex, windex;
	int t = 0;
	for(int i=0;i<N;i++){
		for(int j=0;j<data[i].T;j++){
			
			Feature* fea = data[i].features[j];
			for(int k=0;k<numLabel;k++){
				
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
	
	for(int i=0;i<numLabel*numLabel;i++){
		fvalue[bi_offset_fv+i] = w[bi_offset_w+i];
	}

	inferMarg();
}


void SeqLabelProblem::inferMarg(){

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
	logZ = 0.0;
	double logf_xy_max;
	for(int i=0;i<N;i++){

		if( !bi_factor_updated && sample_updated.find(i) == sample_updated.end() ){
			continue;
		}

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
	}


	//get normalized marginals
	double** marg;
	for(int i=0;i<N;i++){

		if( !bi_factor_updated && sample_updated.find(i) == sample_updated.end() ){
			continue;
		}

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
	}

	delete[] M;
}

void SeqLabelProblem::grad( vector<int>& act_set, //input
		vector<double>& grad){ //output

	grad.clear();

	//Unigram Feature
	Feature* feature;
	int w_index, w_label;
	int factor_index, fvalue_index, t_label;
	double g, loss_deriv;
	int i, fea_exp_index;
	for(i=0; i<act_set.size() && act_set[i]<bi_offset_w ;i++){

		w_index = act_set[i];
		fea_exp_index = w_index / numLabel ;
		w_label = w_index % numLabel;

		feature = &(data_inv[fea_exp_index]);

		g = 0.0;
		vector<pair<int,double> >::iterator it;
		for(it=feature->begin();it!=feature->end();it++){

			factor_index = (it->first/numLabel);
			t_label = labels[factor_index];

			//loss derivative
			if( w_label == t_label ){
				loss_deriv = marginal[factor_index][w_label] - 1.0;
			}else
				loss_deriv = marginal[factor_index][w_label];

			//gradient
			g += loss_deriv * it->second;
		}
		grad.push_back(g);
	}

	//Bigram Feature
	int nnz_yy = act_set.size()-i;
	for(int j=0;j<nnz_yy;j++)
		grad.push_back(0.0);

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

	//Compute M
	/*vector<pair<int,double> >* M  = new vector<pair<int,double> >[numLabel];
	  double one_over_max = exp(-logf_yy_max)/Z_yy;
	  for(int i=0;i<numLabel;i++){
	  for(int j=0;j<numLabel;j++){
	  if( fabs(Mplus[i]-one_over_max) > 1e-12 )
	  M[i].push_back(make_pair(j,Mplus[i*numLabel+j]-one_over_max));
	  }
	  }*/

	double** fw_inc;
	double** bk;
	for(int n=0;n<N;n++){

		fw_inc = seqMargs[n].forward_inc_msg;
		bk = seqMargs[n].backward_msg;

		for(int t=0;t<data[n].T-1;t++){

			//compute normalization constant of marginal joint P(y_{t},y_{t+1})
			/*double Z = one_over_max; //(fw_inc_sum)(bk_sum) = 1
			  for(int i=0;i<numLabel;i++){
			  vector<pair<int,double> >::iterator it;
			  for(it=M[i].begin();it!=M[i].end();it++){
			  Z += it->second*fw_inc[t][i]*bk[t+1][it->first];
			  }
			  }*/
			double Z = 0.0;
			for(int i=0;i<numLabel;i++){
				for(int j=0;j<numLabel;j++)
					Z += fw_inc[t][i]*bk[t+1][j]*Mplus[i*numLabel+j];
			}

			//compuate grad of bigrams
			int lab1,lab2,w_index;
			double p;
			for(int j=0;j<nnz_yy;j++){
				w_index = act_set[i+j] - bi_offset_w;
				lab1 = w_index / numLabel;
				lab2 = w_index % numLabel;

				p = (fw_inc[t][lab1]*bk[t+1][lab2])*(Mplus[w_index]) / Z;

				if( data[n].labels[t]==lab1 && data[n].labels[t+1]==lab2 ){
					/*cerr << "fw_inc[" << t << "]=" << fw_inc[t][0] << "," << fw_inc[t][1] << endl;
					  cerr << "bk[" << t+1 << "]=" << bk[t+1][0] << "," <<bk[t+1][1] << endl;
					  cerr << "Mp[" << w_index << "]=" << Mplus[w_index] << endl;
					  cerr << "p=" << p << ", (" << lab1 << "," << lab2 << ")" << endl;
					  exit(0);*/
					grad[i+j] += p - 1.0;
				}else{
					grad[i+j] += p;
				}
			}
		}
	}

	delete[] Mplus;
	//delete[] M;
}

void SeqLabelProblem::approx_hii( vector<int>& act_set, vector<double>& hii){

	hii.clear();

	//Unigram Feature
	Feature* feature;
	int w_index, w_label;
	int factor_index, fvalue_index, t_label;
	double h, marg;
	int i, fea_exp_index;
	for(i=0; i<act_set.size() && act_set[i]<bi_offset_w ;i++){

		w_index = act_set[i];
		fea_exp_index = w_index / numLabel ;
		w_label = w_index % numLabel;

		feature = &(data_inv[fea_exp_index]);

		h = 0.0;
		vector<pair<int,double> >::iterator it;
		for(it=feature->begin();it!=feature->end();it++){

			factor_index = (it->first/numLabel);
			t_label = labels[factor_index];

			//loss derivative
			marg = marginal[factor_index][w_label];

			//gradient
			h += marg * (it->second)*(it->second) - (marg * it->second) * (marg*it->second);
		}
		hii.push_back(h);
	}

	//Bigram Feature
	int nnz_yy = act_set.size()-i;
	for(int j=0;j<nnz_yy;j++)
		hii.push_back(0.0);

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

	//Compute M
	/*vector<pair<int,double> >* M  = new vector<pair<int,double> >[numLabel];
	  double one_over_max = exp(-logf_yy_max)/Z_yy;
	  for(int i=0;i<numLabel;i++){
	  for(int j=0;j<numLabel;j++){
	  if( fabs(Mplus[i]-one_over_max) > 1e-12 )
	  M[i].push_back(make_pair(j,Mplus[i*numLabel+j]-one_over_max));
	  }
	  }*/

	double** fw_inc;
	double** bk;
	for(int n=0;n<N;n++){

		fw_inc = seqMargs[n].forward_inc_msg;
		bk = seqMargs[n].backward_msg;

		for(int t=0;t<data[n].T-1;t++){

			//compute normalization constant of marginal joint P(y_{t},y_{t+1})
			/*double Z = one_over_max; //(fw_inc_sum)(bk_sum) = 1
			  for(int i=0;i<numLabel;i++){
			  vector<pair<int,double> >::iterator it;
			  for(it=M[i].begin();it!=M[i].end();it++){
			  Z += it->second*fw_inc[t][i]*bk[t+1][it->first];
			  }
			  }*/
			double Z = 0.0;
			for(int i=0;i<numLabel;i++){
				for(int j=0;j<numLabel;j++)
					Z += fw_inc[t][i]*bk[t+1][j]*Mplus[i*numLabel+j];
			}

			//compuate grad of bigrams
			int lab1,lab2,w_index;
			double p;
			for(int j=0;j<nnz_yy;j++){
				w_index = act_set[i+j] - bi_offset_w;
				lab1 = w_index / numLabel;
				lab2 = w_index % numLabel;

				p = (fw_inc[t][lab1]*bk[t+1][lab2])*(Mplus[w_index]) / Z;

				hii[i+j] += p*1*1 - (p*1)*(p*1);
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
	nll += logZ;
	//cerr << logZ << " - " << log_pot_i << endl;

	return nll;
}

double SeqLabelProblem::error(){

	double error = 0.0;
	for(int i=0;i<N;i++){

		Seq* seq = &(data[i]);
		SeqMarg* seqMarg = &(seqMargs[i]);

		for(int t=0; t<seq->T; t++){
			error += (1.0 - seqMarg->marginal[t][ seq->labels[t] ]);
		}
	}
	
	return error / T_total ;
}

void SeqLabelProblem::readProblem(char* data_file, char* info_file=NULL){

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
	T_total=0;
	while( !fin.eof() ){

		fin.getline(_line,MAX_LINE);
		string line(_line);

		if( line.size() <= 1 && !fin.eof() ){

			if( fin.peek() == -1 )
				break;
			
			data.push_back(Seq());
			seq = &(data.back());
			
		}else if( line.size() <= 1 ){
			break;
		}

		split(line," ", tokens);

		//Get Label
		labels.push_back( atoi( tokens[0].c_str() ) );
		
		//Get Features
		Feature* fea = new Feature();
		for(int j=1;j<tokens.size();j++){
			split(tokens[j],":",iv_pair);
			int index = atoi(iv_pair[0].c_str());
			double value =  atof(iv_pair[1].c_str());
			fea->push_back( make_pair( index, value ) );
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
	if( info_file == NULL ){
		cerr << "exit: Sequence Label Problem needs feature template specified as info file" << endl;
		exit(0);
	}
	ifstream finfo(info_file);

	//get unigram template
	finfo.getline(_line,MAX_LINE);//filter one line
	finfo.getline(_line,MAX_LINE);
	string line(_line);
	split(line," ",tokens);

	for(int i=0;i<tokens.size();i++){
		uni_fea_template.push_back(atoi(tokens[i].c_str()));
	}
	finfo.close();

	d_expand = compute_d_expand( raw_d );
	//feature_expand(...) //not done yet

	/*Build data inverted index
	*/
	build_data_inv();
	/*Build Model Parameter Array
	*/
	bi_offset_w = (1+raw_d) * uni_fea_template.size() * numLabel;
	d =  bi_offset_w  +   numLabel*numLabel;

	/*w = new double[d]; //w will be assigned from outside
	for(int i=0;i<d;i++)
		w[i] = 0.0;*/

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
	
	//moved here from constructor
	numBlocks = raw_d+1+1;
	bi_factor_updated = true;
	for(int i=0;i<N;i++)
		sample_updated.insert(i);
	
	//compute marginal, function value etc.
	inferMarg();
}

void SeqLabelProblem::build_data_inv(){

	//initialize data_inv
	for(int i=0;i<d_expand;i++)
		data_inv.push_back(Feature());

	//scan data, expand feature, add to data_inv
	Feature fea_exp;
	int factor_index=0;
	int fea_index;
	for(int i=0;i<N;i++){
		Seq* seq = &(data[i]);
		for(int t=0;t<seq->T;t++){
			feature_expand( *seq, t, fea_exp );

			for(int j=0;j<fea_exp.size();j++){
				fea_index = fea_exp[j].first;
				data_inv[fea_index].push_back( make_pair(factor_index*numLabel, fea_exp[j].second) );
			}
			factor_index++;
		}
	}
}

int SeqLabelProblem::compute_d_expand(int raw_d){

	return 1 + raw_d * uni_fea_template.size();
}

void SeqLabelProblem::feature_expand(Seq& seq, int t, Feature& fea_exp){

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


vector<int>* SeqLabelProblem::buildBlocks(){
	vector<int>* blks = new vector<int>[numBlocks];
	//unigram feature blocks
	for (int b=0; b< numBlocks-1; b++)
		for (int l=0;l<numLabel;l++)
			blks[b].push_back(b*numLabel + l);
	for (int l=0;l<numLabel*numLabel;l++)
		blks[numBlocks-1].push_back(bi_offset_w+l);
	return blks;
}

void SeqLabelProblem::update_Mesg(vector<int> &act_set, double* delta_w){
	if (act_set.size()==0)
		return;
	vector<pair<int,double> > fv_change;
	compute_fv_change(delta_w,  act_set,fv_change);
	update_fvalue(fv_change, 1.0);    
};


void  SeqLabelProblem::derivatives(vector<int> &act_set, vector<double> &grad, vector<double> &hii){

	grad.clear();
	hii.clear();	
	//Unigram Feature
	Feature* feature;
	int w_index, w_label;
	int factor_index, fvalue_index, t_label;
	double h, marg;
	double g, loss_deriv;
	int i, fea_exp_index;
	for(i=0; i<act_set.size() && act_set[i]<bi_offset_w ;i++){

		w_index = act_set[i];
		fea_exp_index = w_index / numLabel ;
		w_label = w_index % numLabel;

		feature = &(data_inv[fea_exp_index]);

		h = 0.0;
		g = 0.0;
		vector<pair<int,double> >::iterator it;
		for(it=feature->begin();it!=feature->end();it++){

			factor_index = (it->first/numLabel);
			t_label = labels[factor_index];

			marg = marginal[factor_index][w_label];
			h += marg * (it->second)*(it->second) - (marg * it->second) * (marg*it->second);

			//loss derivative
			if( w_label == t_label ){
				loss_deriv = marg - 1.0;
			}else
				loss_deriv = marg;

			//gradient
			g += loss_deriv * it->second;
		}
		grad.push_back(g);
		hii.push_back(h);
	}

	//Bigram Feature
	int nnz_yy = act_set.size()-i;
	for(int j=0;j<nnz_yy;j++){
		grad.push_back(0.0);
		hii.push_back(0.0);
	}

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

	//Compute M
	/*vector<pair<int,double> >* M  = new vector<pair<int,double> >[numLabel];
	  double one_over_max = exp(-logf_yy_max)/Z_yy;
	  for(int i=0;i<numLabel;i++){
	  for(int j=0;j<numLabel;j++){
	  if( fabs(Mplus[i]-one_over_max) > 1e-12 )
	  M[i].push_back(make_pair(j,Mplus[i*numLabel+j]-one_over_max));
	  }
	  }*/

	double** fw_inc;
	double** bk;
	for(int n=0;n<N;n++){

		fw_inc = seqMargs[n].forward_inc_msg;
		bk = seqMargs[n].backward_msg;

		for(int t=0;t<data[n].T-1;t++){

			//compute normalization constant of marginal joint P(y_{t},y_{t+1})
			/*double Z = one_over_max; //(fw_inc_sum)(bk_sum) = 1
			  for(int i=0;i<numLabel;i++){
			  vector<pair<int,double> >::iterator it;
			  for(it=M[i].begin();it!=M[i].end();it++){
			  Z += it->second*fw_inc[t][i]*bk[t+1][it->first];
			  }
			  }*/
			double Z = 0.0;
			for(int i=0;i<numLabel;i++){
				for(int j=0;j<numLabel;j++)
					Z += fw_inc[t][i]*bk[t+1][j]*Mplus[i*numLabel+j];
			}

			//compuate grad of bigrams
			int lab1,lab2,w_index;
			double p;
			for(int j=0;j<nnz_yy;j++){
				w_index = act_set[i+j] - bi_offset_w;
				lab1 = w_index / numLabel;
				lab2 = w_index % numLabel;

				p = (fw_inc[t][lab1]*bk[t+1][lab2])*(Mplus[w_index]) / Z;
				hii[i+j] += p*1*1 - (p*1)*(p*1);

				if( data[n].labels[t]==lab1 && data[n].labels[t+1]==lab2 ){
					/*cerr << "fw_inc[" << t << "]=" << fw_inc[t][0] << "," << fw_inc[t][1] << endl;
					  cerr << "bk[" << t+1 << "]=" << bk[t+1][0] << "," <<bk[t+1][1] << endl;
					  cerr << "Mp[" << w_index << "]=" << Mplus[w_index] << endl;
					  cerr << "p=" << p << ", (" << lab1 << "," << lab2 << ")" << endl;
					  exit(0);*/
					grad[i+j] += p - 1.0;
				}else{
					grad[i+j] += p;
				}
			}
		}
	}

	delete[] Mplus;

}


ostream& operator<<(ostream& out, Feature& vect){
	
	int size = vect.size();
	out.write( (char*) &size, sizeof(int));
	for(int i=0;i<size;i++){
		out.write( (char*) &(vect[i].first), sizeof(int) );
		out.write( (char*) &(vect[i].second), sizeof(double) );
	}
}

istream& operator>>(istream& in, Feature& vect){
	
	vect.clear();
	
	int size;
	in.read((char*) &size, sizeof(int));
	vect.resize( size );
	for(int i=0;i<size;i++){
		
		in.read( (char*) &(vect[i].first), sizeof(int));
		in.read( (char*) &(vect[i].second), sizeof(double));
	}
}

ostream& operator<<(ostream& out, vector<Feature*>& vect){
	
	int size = vect.size();
	out.write( (char*) &size, sizeof(int));
	for(int i=0;i<size;i++){
		Feature* fea = vect[i];
		out << (*fea) ;
	}
}

istream& operator>>(istream& in, vector<Feature*>& vect){
	
	for(int i=0;i<vect.size();i++){
		if(vect[i]!=NULL)delete vect[i];
	}
	vect.clear();
	
	int size;
	in.read( (char*)&size, sizeof(int) );
	for(int i=0;i<size;i++){
		Feature* fea = new Feature();
		in >> (*fea);
		vect.push_back(fea);
	}
}

ostream& operator<<(ostream& out, vector<Feature>& vect){
	
	int size = vect.size();
	out.write( (char*) &size, sizeof(int));
	for(int i=0;i<size;i++){
		out << (vect[i]) ;
	}
}

istream& operator>>(istream& in, vector<Feature>& vect){
	
	int size;
	in.read( (char*)&size, sizeof(int) );
	vect.resize(size);
	for(int i=0;i<size;i++){
		in >> (vect[i]);
	}
}

ofstream& operator<<(ofstream& os, map<int,string>& m){
	
	int size = m.size();
	os.write( (char*) &(size),  sizeof(int) );
	map<int,string>::iterator it;
	for(it=m.begin(); it!=m.end(); it++){
		os.write( (char*) &(it->first), sizeof(int));
		os.write( (char*) &(it->second), sizeof(string));
	}
	
	return os;
}

ifstream& operator>>(ifstream& is, map<int,string>& m){
	
	m.clear();
	
	int size;
	is.read( (char*) &(size),  sizeof(int) );
	int tmp1;
	string tmp2;
	for(int i=0;i<size;i++){
		
		is.read( (char*) &(tmp1), sizeof(int) );
		is.read( (char*) &(tmp2), sizeof(string) );
		
		m.insert(make_pair(tmp1,tmp2));
	}
	
	return is;
}

ofstream& operator<<(ofstream& os, map<string,int>& m){
	
	int size = m.size();
	os.write( (char*) &(size),  sizeof(int) );
	map<string,int>::iterator it;
	for(it=m.begin(); it!=m.end(); it++){
		os.write( (char*) &(it->first), sizeof(string));
		os.write( (char*) &(it->second), sizeof(int));
	}
	
	return os;
}

ifstream& operator>>(ifstream& is, map<string,int>& m){
	
	m.clear();
	
	int size;
	is.read( (char*) &(size),  sizeof(int) );
	string tmp1;
	int tmp2;
	for(int i=0;i<size;i++){
		
		is.read( (char*) &(tmp1), sizeof(string) );
		is.read( (char*) &(tmp2), sizeof(int) );
		
		m.insert(make_pair(tmp1,tmp2));
	}
	
	return is;
}

ofstream& operator<<(ofstream& os, vector<int>& vect){
	
	int size = vect.size();
	os.write( (char*) &size, sizeof(int) );
	for(int i=0;i<size;i++)
		os.write( (char*) &(vect[i]), sizeof(int) );
	
	return os;
}

ifstream& operator>>(ifstream& is, vector<int>& vect){
	
	int size;
	is.read( (char*) &size, sizeof(int) );
	vect.resize(size);
	for(int i=0;i<size;i++)
		is.read( (char*) &(vect[i]), sizeof(int) );
	
	return is;
}

ofstream& operator<<(ofstream& os, set<int>& s){
	
	int size = s.size();
	os.write( (char*) &size, sizeof(int) );
	set<int>::iterator it;
	for(it=s.begin();it!=s.end();it++)
		os.write( (char*)&(*it), sizeof(int) );
	
	return os;
}

ifstream& operator>>(ifstream& is, set<int>& s){
	
	int size;
	is.read( (char*) &size, sizeof(int) );
	s.clear();
	for(int i=0;i<size;i++){
		int tmp;
		is.read( (char*) &tmp, sizeof(int) );
		s.insert(tmp);
	}
	
	return is;
}


void SeqLabelProblem::save(char* filename){
	
	ofstream fout(filename, ios::out|ios::binary);
	
	//save abstract problem
	fout.write( (char*) &N, sizeof(int));
	fout.write( (char*) &d, sizeof(int));
	fout.write( (char*) &n, sizeof(int));
	fout.write( (char*) &fvalue, sizeof(double)*n);
	fout.write( (char*) &numBlocks, sizeof(int) );
	
	//seq_label_problem begins here
	fout.write( (char*) &raw_d, sizeof(int) );
	fout.write( (char*) &T_total, sizeof(int) );
	fout.write( (char*) &numLabel, sizeof(int));
	
	fout.write( (char*) &bi_offset_w, sizeof(int) );
	fout.write( (char*) &bi_offset_fv, sizeof(int) );
	
	//save data
	fout << labels;
	
	for(int i=0;i<N;i++){
		
		fout << data[i].features;
	}
	
	//save data_inv
	fout << data_inv ;
	
	fout << sample_updated;
	fout.write( (char*) &bi_factor_updated, sizeof(bool));
	
	//messages
	for(int i=0;i<T_total;i++)
		fout.write( (char*) forward_msg[i], sizeof(double)*numLabel );
	for(int i=0;i<T_total;i++)
		fout.write( (char*) backward_msg[i], sizeof(double)*numLabel );
	for(int i=0;i<T_total;i++)
		fout.write( (char*) forward_inc_msg[i], sizeof(double)*numLabel );
	for(int i=0;i<T_total;i++)
		fout.write( (char*) marginal[i], sizeof(double)*numLabel );
	
	fout.write( (char*) &logZ, sizeof(double) );

	//feature template
	fout << uni_fea_template;
	fout.write( (char*) &d_expand, sizeof(int) );
	
	fout.close();
}


void SeqLabelProblem::load(char* filename){
	
	ifstream fin(filename, ios::out|ios::binary);
	
	//load abstract problem
	fin.read( (char*) &N, sizeof(int));
	fin.read( (char*) &d, sizeof(int)); 
	fin.read( (char*) &n, sizeof(int));
	fvalue = new double[n];
	fin.read( (char*) fvalue, sizeof(double)*n);
	fin.read( (char*) &numBlocks, sizeof(int) );
	
	//seq_label_problem begins here
	fin.read( (char*) &raw_d, sizeof(int) ); //should be assigned from outside
	fin.read( (char*) &T_total, sizeof(int) );
	fin.read( (char*) &numLabel, sizeof(int)); //should be assigned from outside
	
	fin.read( (char*) &bi_offset_w, sizeof(int) );
	fin.read( (char*) &bi_offset_fv, sizeof(int) );
	
	//load data
	fin >> labels;
	data.resize(N);
	int k=0;
	for(int i=0;i<N;i++){
		data[i].labels = &(labels[k]);
		fin >> data[i].features;
		data[i].T = data[i].features.size();
		
		k += data[i].T;
	}
	
	//load data_inv
	fin >> data_inv ;
	
	fin >> sample_updated;
	fin.read( (char*) &bi_factor_updated, sizeof(bool));
	
	//build factors pointers
	k=0;
	if(factors==NULL){
		factors = new SeqFactor[N];
		for(int i=0;i<N;i++)
			factors[i].factor_xy = new double*[data[i].T];
	}
	for(int i=0;i<N;i++){
		for(int t=0;t<data[i].T;t++){

			factors[i].factor_xy[t] = &(fvalue[k]);
			k += numLabel;
		}
	}
	//Bigram 
	factor_yy = &(fvalue[k]);

	//messages
	if(forward_msg==NULL){
		forward_msg = new double*[T_total];
		backward_msg = new double*[T_total];
		forward_inc_msg = new double*[T_total];
		marginal = new double*[T_total];
		for(int i=0;i<T_total;i++){
			forward_msg[i] = new double[numLabel];
		}
		for(int i=0;i<T_total;i++){
			backward_msg[i] = new double[numLabel];
		}
		for(int i=0;i<T_total;i++){
			forward_inc_msg[i] = new double[numLabel];
		}
		for(int i=0;i<T_total;i++){
			marginal[i] = new double[numLabel];
		}
	}

	for(int i=0;i<T_total;i++){
		fin.read( (char*) forward_msg[i], sizeof(double)*numLabel );
	}
	for(int i=0;i<T_total;i++){
		fin.read( (char*) backward_msg[i], sizeof(double)*numLabel );
	}
	for(int i=0;i<T_total;i++){
		fin.read( (char*) forward_inc_msg[i], sizeof(double)*numLabel );
	}
	for(int i=0;i<T_total;i++){
		fin.read( (char*) marginal[i], sizeof(double)*numLabel );
	}
	fin.read( (char*) &logZ, sizeof(double) );
	
	//build messages pointers
	k = 0;
	if(seqMargs==NULL)
		seqMargs = new SeqMarg[N];
	for(int i=0;i<N;i++){

		seqMargs[i].forward_msg = &(forward_msg[k]);
		seqMargs[i].forward_inc_msg = &(forward_inc_msg[k]);
		seqMargs[i].backward_msg = &(backward_msg[k]);
		seqMargs[i].marginal = &(marginal[k]);

		k += data[i].T;
	}
	
	
	//feature template
	fin >> uni_fea_template;
	fin.read( (char*)&d_expand, sizeof(int) );
}

void SeqLabelProblem::release(){
	
	//release data
	for(int i=0;i<N;i++){
		for(int j=0;j<data[i].T;j++){
			delete data[i].features[j];
			data[i].features[j] = NULL;
		}
	}

	//release data_inv
	data_inv.clear();
}
