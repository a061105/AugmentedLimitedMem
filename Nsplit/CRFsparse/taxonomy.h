#ifndef TAXONOMY
#define TAXONOMY

#include "HashTable.h"
#include "problem.h"
#include <map>
//#include <set>

typedef vector<pair<int,double> > Feature;
typedef vector<pair<int,double> > Sample;
class catagory{
	public:
		catagory(){};
		int ind; //ind in the catArray
		//int numChild;
		bool isleaf;
		vector<int> childInd; //children's index in catArray
		vector<catagory* > childrenAddrArray;
		catagory* parent;
		int parInd; //parent's index in catArray;
		//int *isLabeled; //N 
		//int startInd; //starting index in w array
		bool addChild(catagory* cadd){
			childrenAddrArray.push_back(cadd);
		}
};

class TaxonomyProblem:public Problem{
	/* w: d *|Y|, |Y| is the number of total catagories
	 *
	 * fvalue: N*|Y|
	 */

	public:
		TaxonomyProblem(char* data_file, char* info_file);

		//Functions for "Problem"
		void compute_fv_change(double* w_change, vector<int>& act_set, //input
				vector<pair<int,double> >& fv_change); //output
		
		void compute_fvalue();

		void update_fvalue(vector<pair<int,double> >& fv_change, double scalar);

		void grad( vector<int>& act_set, //input
				vector<double>& g);//output
		double fun();
		void downMesgPass(int ind,int nn);

		void update_Mesg(vector<int> &act_set, double* delta_w);

		//    double firstDerivative(int coordinate);

		//double secondDerivative(int coordinate);

		void readProblem(char* data_file, char* info_file);

		void save(char* filename);
		void load(char* filename);

		//Function specific to Sequence Labeling Problem
		TaxonomyProblem(char* data_file);
		// virtual ~TaxonomyProblem();
		void buildCatArray(catagory* curCat);
		void infer();
		void infer(int block);
		double upMesgPass(catagory* curCat,int nn);
		void  derivatives(vector<int> &act_set, vector<double> &firstDerivative, vector<double> &secondDerivative);
		vector<int>* buildBlocks();
		double errorrate;
		int L; //number of label(leaf)
		int K; //number of total catagories(include inner node and leaf)
		int raw_d; //number of features
		int nonleafInd;
		int leafInd;
		catagory *root;
		catagory** catArray;
		double *downMesg; //n
		double *upMesg; //n
		double *fv_change_table;
		double *labelMesg;//n
		vector<Feature > features; //d*N
		vector<int> labels;
		//vector<vector<double> > condProb; //K*N

		//vector<vector<int> > ispath; //K*N
		//vector<catagory> catagories;

		TaxonomyProblem(){
			errorrate = 0.0;
		}
}; 

#endif
