#ifndef DD_AUG
#define DD_AUG

#include "Subsolver.h"

class DDAugmented:public Augmented{
	
	public:

	DDAugmented();
	DDAugmented(double* _w_init);

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
};

#endif
