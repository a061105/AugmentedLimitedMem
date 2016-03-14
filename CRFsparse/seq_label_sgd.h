#ifndef SEQ_LABEL_SGD
#define SEQ_LABEL_SGD

#include "HashTable.h"
#include "problem.h"
#include "seq_label.h"
#include <map>


class SeqLabelProblem_SGD:public Problem_SGD{
	
	/* w: d_expand *|Y| + d_expand_2*|Y|*|Y|
	 *
	 * fvalue: N*T*|Y| + |Y|*|Y| (no bi-fea)
	 * 	   N*T*|Y| +  NT*|Y|*|Y| (with bi-fea)
	 */

	public:
	int raw_d;
	double* fvalue;
	int n;

	//Functions for "Problem"
	void applyGrad( int data_index, //input
			            double eta, SGD* sgd);//output

	double fun();
	void readProblem(char* data_file);
	
	//Function specific to Sequence Labeling Problem
	SeqLabelProblem_SGD(char* data_file);
	
	int numLabel;
	map<string,int> label_index_map;
	map<int,string> label_name_map;
	
	//private:
	
	int bi_offset_w;
	int bi_offset_fv;
	
	//store sequences 
	vector<Seq> data; //N
	vector<int> labels; //T_total
	
	//factor values
	SeqFactor* factors; //N
	double* factor_yy;
	
	//calibrated forward (exlusive), backward (inclusive) messages
	SeqMarg* seqMargs; // N
	double inferMarg(int data_index);
	
	double** forward_msg; //T_total * |Y|
	double** backward_msg; 
	double** forward_inc_msg; 
	double** marginal;
	double logZ_all;
	
	//feature template
	vector<int> uni_fea_template;
	int d_expand;
	int compute_d_expand(int raw_d);
	void feature_expand(Seq& seq, int t, Feature& fea_exp);
};

#endif
