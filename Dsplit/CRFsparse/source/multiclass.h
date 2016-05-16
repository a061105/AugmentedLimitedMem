#ifndef MULTICLASS
#define MULTICLASS

#include <set>
#include "problem.h"
#include <map>
#include "util.h"

class MulticlassProblem:public Problem{

	/* w: d *|Y|
	 *
	 * fvalue: N*|Y|
	 */

	public:
		//Int raw_d;

		//Functions for "Problem"
		void compute_fv_change(double* w_change, vector<Int>& act_set, pair<Int,Int> range, //input
				vector<pair<Int,double> >& fv_change); //output

		void update_fvalue(vector<pair<Int,double> >& fv_change, double scalar);

		void grad( vector<Int>& act_set, pair<Int,Int> range, //input
				vector<double>& g);//output

		double fun();
		double fun(vector<Int>& act_set);
		double fun(Int numBlock, vector< pair<Int,Int> > range, Swapper* swapper);

		void test_accuracy(const char* output_file);

		double train_accuracy();

		void readProblem(char* data_file);
		void readProblem(char* model_file, char* data_file);

		//Function specific to Sequence Labeling Problem
		MulticlassProblem(char* data_file);
		MulticlassProblem(char* model_file, char* data_file);

		//Int K; //#labels
		map<string,Int> label_index_map;
		map<Int,string> label_name_map;
		
		vector<Feature*> data; //N
		vector<Int> labels;    //N

		vector<Feature> data_inv; //for each fea j, store set factors(j): the affected factors (n,t) by fea j
    		set<Int> sample_updated;
		void build_data_inv();
		
		//factor values
		double** factor_xy; //N*|Y| table"
		void compute_fvalue();
		
		//calibrated forward (exlusive), backward (inclusive) messages
		double** marginal; // N by K
		double logZ;
		void inferMarg();

};

#endif
