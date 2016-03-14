#ifndef BCD_H
#define BCD_H
#include "problem.h"
#include "optimizer.h"
//extern class Problem_CD;

class BCD:public optimizer{
    public:
        void minimize(Problem *prob);
};

#endif
