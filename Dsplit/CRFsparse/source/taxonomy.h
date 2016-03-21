#ifndef TAXONOMY
#define TAXONOMY

#include "HashTable.h"
#include "problem.h"
#include <map>

typedef vector<pair<Int,double> > Feature;
typedef vector<pair<Int,double> > Sample;

class category{
    public:
        category(){};
        Int ind; //index in the catArray
        Int raw_label;
        bool isleaf; //true if this category is leaf
        vector<Int> childInd; //children's index in catArray
        vector<category* > childrenAddrArray;
        category* parent;
        Int parInd; //parent's index in catArray;
        void addChild(category* cadd){
           childrenAddrArray.push_back(cadd);
        }
};

class TaxonomyProblem:public Problem{
   	/* w: d *|Y|, |Y| is the number of total categories
	 *
	 * #fvalue: N*|Y|
	 */

	public:
	
	//Functions for "Problem"
	void compute_fv_change(double* w_change, vector<Int>& act_set, //input
			vector<pair<Int,double> >& fv_change); //output
	
	void update_fvalue(vector<pair<Int,double> >& fv_change, double scalar);
	
	void grad( vector<Int>& act_set, //input
			vector<double>& g);//output
	
    double fun();
    
    double train_accuracy();

    void test_accuracy(const char* output_file);

    void readProblem(char* data_file);
    void readProblem(char* model_file, char* data_file);
    
    //Functions and variables specific to Sequence Labeling Problem
    Int L; //number of label(leaf)
    Int K; //number of total categories(include inner nodes and leaf)
    //Int raw_d; //number of features
    
    category *root; // root of the taxonomy tree
    category** catArray; // an array of category poInts
    Int nonleafInd; //category index in catArray for nonleaf categories
    Int leafInd;    //category index in catArray for leaf categories
    double *downMesg; //size n
    double *upMesg; //size n
    double *fv_change_table;
    double *labelMesg;//size n, denote if a category is an ancester(including itself) of the observed category
    vector<Feature > features; //d*N matrix
    vector<Int> labels;

	TaxonomyProblem(char* data_file);
	TaxonomyProblem(char* model_file,char* data_file);

    void buildCatArray(category* curCat); //build catArray
    void infer(); //inference for whole data set
    double upMesgPass(category* curCat,Int nn);
    void downMesgPass(Int ind,Int nn);
}; 

#endif
