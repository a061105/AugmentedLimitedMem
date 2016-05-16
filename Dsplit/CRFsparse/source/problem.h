#ifndef PROBLEM
#define PROBLEM

#include<vector>
#include"util.h"
#include "Swapper.h"

using namespace std;

/** This is a Interface for problem aimed to utilize
 *  our optimization package. 
 */

class Problem{
	
	public:
		Int N; // number of samples
		Int d; //number of features
		Int db;
		Int K;
		ValueMap w; // weights, d by 1 array, d is number of features (defined by template)
		Int n; //number of factors
		ValueMap fvalue; // factor value, n by 1 array, n is #factor. 
		Int raw_d; //number of raw features
		
		/* Here is the definition of factor values and the reason we use it.
		 * For log-linear models, factors are Intermediate variables 
		 * for calculating gradients and function values from data and weights.
		 * Once they are computed from weights and data, they can be used to 
		 * calculate gradients and function values without involving the data. 
		 * For example, the factors in the classification with taxonomy problem
		 * are inner product of <W_k, x>, where W_k is the weights for k-th class 
		 * and x is an instance of data set. 
		 * A linear change of the weights, say alpha * w, will lead to a linear 
		 * change of factors, alpha * fvalue. Considering the line search 
		 * procedure, w_new = w + alpha * w_change, where alpha will probably 
		 * try over multiple values and in each try, the function value is 
		 * calculated. If we record fvalue, we don't need to calculate <W_k,x> 
		 * in each try. Just calculate function value from new fvalue. 
		 */
		//calculate factor value change due to a descent direction, w_change.
		virtual void compute_fv_change(ValueMap w_change, vector<Int>& act_set, pair<Int,Int> range, //input
				vector<pair<Int,double> >& fv_change)=0; //output

		//update new fvalue = fvalue + scalar * fv_change
		virtual void update_fvalue(vector<pair<Int,double> >& fv_change, double scalar)=0;

		virtual void grad( vector<Int>& act_set, pair<Int,Int> range, //input
				vector<double>& g)=0;//output

		virtual double fun()=0;
		virtual double fun(vector<Int>& act_set) = 0;
		virtual double fun(Int numBlock, vector< pair<Int,Int> > range, Swapper* swapper) = 0;

		virtual double train_accuracy() = 0;
		virtual void test_accuracy(const char* output_file) = 0; //output prediction value to a file while computing acc

		virtual void readProblem(char* data_file)=0;
		virtual void readProblem(char* model_file, char* data_file)=0; //used for testing or training with initial model
};


class Param{
	public:
		Param(){
			problem_type = 1;
			solver = 0;
			info_file = NULL;
			max_iter = 100;
			lambda = 1.0;
			theta  = 0.0;
			theta2 = 0.0;
			epsilon = 1e-2;
			predict_method = 1;
		}
		Int problem_type;
		Int solver;
		char* info_file;
		Int max_iter;
		double lambda;
		double theta;
		double theta2;
		double epsilon; //termination criterion
		char* model_file;

		Int predict_method;
};

extern Param param;

#endif
