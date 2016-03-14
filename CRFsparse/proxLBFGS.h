#ifndef PROXLBFGS
#define PROXLBFGS
#include"optimizer.h"

class proxLBFGS:public optimizer{
    public:
        proxLBFGS();
        ~proxLBFGS();
        void minimize(Problem *prob);
}; 

#endif
