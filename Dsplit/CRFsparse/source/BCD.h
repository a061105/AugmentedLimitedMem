#ifndef BCD
#define BCD
#include"optimizer.h"

class bCD:public optimizer{
    public:
//        proxQN();
//        ~proxQN();
        void minimize(Problem *prob);
	void readmodel();
	void wirtemodel();
}; 

#endif
