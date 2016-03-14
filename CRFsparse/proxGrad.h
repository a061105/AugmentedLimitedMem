#ifndef PROXGRAD
#define PROXGRAD
#include "optimizer.h"

class proxGrad:public optimizer{
    public:
        proxGrad();
        ~proxGrad();
        void minimize(Problem *prob);
};

#endif
