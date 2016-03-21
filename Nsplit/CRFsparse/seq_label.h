#ifndef SEQ_LABEL
#define SEQ_LABEL

#include "HashTable.h"
#include "problem.h"
#include <map>

ostream& operator<<(ostream& out, Feature& vect);
istream& operator>>(istream& in, Feature& vect);

ostream& operator<<(ostream& out, vector<Feature*>& vect);
istream& operator>>(istream& in, vector<Feature*>& vect);

ostream& operator<<(ostream& out, vector<Feature>& vect);
istream& operator>>(istream& in, vector<Feature>& vect);

ostream& operator<<(ostream& out, map<int,string>& m);
istream& operator>>(istream& in, map<int,string>& m);

ostream& operator<<(ostream& out, map<string,int>& m);
istream& operator>>(istream& in, map<string,int>& m);

ostream& operator<<(ostream& out, vector<int>& vect);
istream& operator>>(istream& in, vector<int>& vect);

ostream& operator<<(ostream& out, set<int>& vect);
istream& operator>>(istream& in, set<int>& vect);

class Seq{
	public:
		Seq(){ T = 0; }
		int T;
		int* labels; //1*T virtual
		vector<Feature*> features; // 1*T
};

class SeqFactor{
	public:
		double** factor_xy; //1*T array of "1*|Y| table" (virtual)
};

class SeqMarg{ //virtual
	public:
		double** forward_msg; //(exclusive)
		double** forward_inc_msg; //(incusive)
		double** backward_msg; //(inclusive)
		double** marginal;
};

class SeqLabelProblem:public Problem{

	/* w: d_expand *|Y| + d_expand_2*|Y|*|Y|
	 *
	 * fvalue: N*T*|Y| + |Y|*|Y| (no bi-fea)
	 * 	   N*T*|Y| +  NT*|Y|*|Y| (with bi-fea)
	 */

	public:
		int raw_d;
		int T_total;

		//Functions for "Problem"
		SeqLabelProblem(int _raw_d, int L);
		
		void compute_fv_change(double* w_change, vector<int>& act_set, //input
				vector<pair<int,double> >& fv_change); //output
		
		void compute_fvalue();

		void update_fvalue(vector<pair<int,double> >& fv_change, double scalar);

		void grad( vector<int>& act_set, //input
				vector<double>& g);//output

		void approx_hii( vector<int>& f_index,
				vector<double>& hii);

		double fun();
		double error();

		void readProblem(char* data_file, char* info_file);
		void inferMarg(int fea_exp_index); 
		void update_Mesg(vector<int> &act_set, double* delta_w);
		vector<int>* buildBlocks();
		void  derivatives(vector<int> &act_set, vector<double> &firstDerivative, vector<double> &secondDerivative);
		//Function specific to Sequence Labeling Problem
		SeqLabelProblem(char* data_file);
		SeqLabelProblem();

		int numLabel;
		
		void save(char* filename); //save "data" and "data_inv" into disk in binary format
		void load(char* filename); //load "data" and "data_inv" from disk into memory
		void release(); //release "data" and "data_inv" from memory
		
		//private:

		int bi_offset_w;
		int bi_offset_fv;
		
		//store sequences 
		vector<Seq> data; //#  N
		vector<int> labels; //T_total
		vector<Feature> data_inv; //# for each fea j, store set factors(j): the affected factors (n,t) by fea j
		void build_data_inv();
		set<int> sample_updated;
		bool bi_factor_updated;
		//factor values
		SeqFactor* factors; //N
		double* factor_yy;
		
		//calibrated forward (exlusive), backward (inclusive) messages
		SeqMarg* seqMargs; // N
		void inferMarg();

		double** forward_msg; //T_total * |Y|
		double** backward_msg; 
		double** forward_inc_msg; 
		double** marginal;
		double logZ;

		//feature template
		vector<int> uni_fea_template;
		int d_expand;
		int compute_d_expand(int raw_d);
		void feature_expand(Seq& seq, int t, Feature& fea_exp);
};

#endif
