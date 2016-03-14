#ifndef PROBLEM
#define PROBLEM

#include<vector>
#include"util.h"
#include"SGD.h"
#include "../Subsolver.h"
//#include"CD.h"
using namespace std;

const int MAX_LINE = 100000;
const int FNAME_LENGTH = 1000;

//extern class CD;
class SGD;
/** This is a interface for problem aimed to utilize
 *  our optimization package
 */
class Problem{

	public:
		int N; // number of samples
		int d; //number of features
		
		double* w; // d by 1, d is number of features (defined by template)
		/*double* u; // dual variable from ADMM
	        double* z; // consensus variable from ADMM
		double rho; // ADMM parameter*/
		Augmented* augmented;

		int n; //number of factors
		double* fvalue; // n by 1, n is #factor
		int numBlocks;	
		virtual void compute_fv_change(double* w_change, vector<int>& act_set, //input
				vector<pair<int,double> >& fv_change)=0; //output

		virtual void update_fvalue(vector<pair<int,double> >& fv_change, double scalar)=0;
		
		virtual void compute_fvalue()=0;

		virtual void grad( vector<int>& act_set, //input
				vector<double>& g)=0;//output

		virtual double fun()=0;

		virtual void readProblem(char* data_file, char* info_file)=0;

		double Eval(vector<double> &inputw, vector<double> & g);
		vector<int> wholeset;  //used for OWLQN
		//used for coordinate descent
		//virtual void update_fvalue(double w_change, int coordinate)=0;

		//virtual double firstDerivative(int coordinate)=0;
		virtual void  derivatives(vector<int> &act_set, vector<double> &firstDerivative,         vector<double> &secondDerivative)=0;
		virtual void update_Mesg(vector<int> &act_set, double* delta_w)=0;
		virtual vector<int>* buildBlocks()=0;

		virtual void save(char*  fileName)=0;
		virtual void load(char* fileName)=0;
		virtual void release()=0;
};

class Problem_SGD{

	public:
		int d; //number of features
		double* w; // d by 1, d is number of features (defined by template)
		int N; // number of samples

		virtual void applyGrad( int data_index, //input
				double eta, SGD* sgd)=0;//output

		virtual double fun()=0;

		virtual void readProblem(char* data_file)=0;
};
/*
   class Problem_CD{

   public:
   int N; // number of samples
   int d; //number of features
   double* w; // d by 1, d is number of features (defined by template)
   int n; //number of factors
   double* fvalue; // n by 1, n is #factor


   virtual double fun()=0;

   virtual void readProblem(char* data_file)=0;
   };
   */
class Param{
	public:
		Param(){
			problem_type = 2;
			solver = 0;
			info_file = NULL;
			max_iter = 100;
			lambda = 1.0;
			epsilon = 1e-6;
			eta0 = 1e-2;
			alpha = 0.9;
		}
		int problem_type;
		int solver;
		char* info_file;
		int max_iter;
		double lambda;
		double epsilon;
		double eta0;
		double alpha;
};

extern Param param;

#endif
