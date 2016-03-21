#ifndef TAXONOMY_SGD
#define TAXONOMY_SGD

#include "HashTable.h"
#include "problem.h"
#include "taxonomy.h"
#include <map>

class TaxonomyProblem_SGD:public Problem_SGD{
   	/* w: d *|Y|, |Y| is the number of total catagories
	 *
	 * fvalue: N*|Y|
	 */

	public:
    int raw_d;
	double* fvalue;
	int n;

	
	//Functions for "Problem"
	void applyGrad(int data_index,//input
            double eta,SGD* sgd);//output

    double fun();
    
    //void update_Mesg(double *w_change, int f);

//    double firstDerivative(int coordinate);
	
    //double secondDerivative(int coordinate);

    void readProblem(char* data_file);

    //Function specific to Sequence Labeling Problem
	TaxonomyProblem_SGD(char* data_file);
    void buildCatArray(catagory* curCat);
    void infer();
    double upMesgPass(catagory* curCat,int nn);
    double errorrate ;
    int L; //number of label(leaf)
    int K; //number of total catagories(include inner node and leaf)
    int nonleafInd;
    int leafInd;
    catagory *root;
    catagory** catArray;
    double *downMesg; //n
    double *upMesg; //n
    double *fv_change_table;
    int *labelMesg;//n
    vector<Sample > data; //d*N
    vector<int> labels;
    
    TaxonomyProblem_SGD(){
	errorrate = 0.0;
    }

}; 

#endif
