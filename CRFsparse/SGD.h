#ifndef SGD_H
#define SGD_H
#include "problem.h"

class Problem_SGD;

class SGD{
    public:
        double lambda;
        int max_iter;
        double epsilon;
        double eta0;
        double alpha;
        double *q;
        double u;
        void penalty(int j,double *w);
        void minimize(Problem_SGD *prob);

	SGD(){
        	max_iter = 100;
        	epsilon = 1e-6;
        	eta0 = 1e-2;
        	alpha = 0.9;
	}
};

#endif

