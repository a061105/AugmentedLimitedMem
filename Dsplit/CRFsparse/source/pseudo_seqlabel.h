#ifndef PSEUDO_SEQLABEL
#define PSEUDO_SEQLABEL

#include <set>
#include "problem.h"
#include <map>
#include "util.h"

class PseudoSeqLabelProblem:public Problem{
	
	/* w: d *|Y| + |Y|*|Y|
	 *
	 * fvalue: N*|Y|
	 */

	public:
		Int w_bi_offset;

		//Functions for "Problem"
		void compute_fv_change(double* w_change, vector<Int>& act_set, //input
				vector<pair<Int,double> >& fv_change); //output

		void update_fvalue(vector<pair<Int,double> >& fv_change, double scalar);

		void grad( vector<Int>& act_set, //input
				vector<double>& g);//output
		
		double collect_grad( Int w_label, Feature* fea );

		double fun();
		double fun(vector<Int>& act_set);

		void test_accuracy(const char* output_file);

		double train_accuracy();

		void readProblem(char* data_file);
		void readProblem(char* model_file, char* data_file);

		//Function specific to Sequence Labeling Problem
		PseudoSeqLabelProblem(char* data_file);
		PseudoSeqLabelProblem(char* model_file, char* data_file);

		Int K; //#labels
		map<string,Int> label_index_map;
		map<Int,string> label_name_map;
		
		vector<Feature*> data; //N
		vector<Feature*> data_bi_left;
		vector<Feature*> data_bi_right;
		vector<Int> labels;    //N
		
		vector<Feature> data_inv; //for each fea j, store set factors(j): the affected factors (n,t) by fea j
    		vector<Feature> bi_left_inv;
    		vector<Feature> bi_right_inv;
		
		void build_data_inv();
		
		//factor values
		set<Int> sample_updated;
		double** factor_xy; //N*|Y| table
		void compute_fvalue();

		//calibrated forward (exlusive), backward (inclusive) messages
		double** marginal; // N by K
		double logZ;
		void inferMarg();
};

#endif
