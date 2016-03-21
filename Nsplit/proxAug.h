#ifndef prox_AUG
#define prox_AUG

#include "Subsolver.h"

class proxAugmented:public Augmented{
	
	public:

	proxAugmented();

	virtual double fun(double* w);
	virtual void grad(double* w, double* g);
	virtual void Hs(double* w, double* Hs);
	//coordinate-wise
	virtual double fun(double w, int index);
	virtual double grad(double w, int index);
	virtual double Hii(double w, int index);

	virtual double* getModel();
	virtual double* getMu_B();
	virtual void getAlpha(double* alpha);
	virtual void setAlpha(double* alpha);

	double* mu;
	double* w_init;
	double* global_alpha;
	std::vector<int> instance;
	double eta_t;
	double* wt;
};

#endif
