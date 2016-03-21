#ifndef OPTIMIZER
#define OPTIMIZER

#include "problem.h"
using namespace std;

class optimizer{
    public:
        double lambda;
	Int max_iter;
        double epsilon;
        virtual void minimize(Problem *prob) = 0;
};

#endif
