#ifndef OPTIMIZER
#define OPTIMIZER

#include "problem.h"
using namespace std;

class optimizer{
    public:
        double lambda;
        int max_iter ;
        double epsilon ;
        virtual void minimize(Problem *prob) = 0;

	optimizer(){
		max_iter = 100;
		epsilon = 1e-6;
	}
};

#endif
