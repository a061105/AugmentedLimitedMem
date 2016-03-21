#ifndef ADMM_AUG
#define ADMM_AUG

#include "Subsolver.h"

class ADMMAugmented:public Augmented{
	
	public:
	ADMMAugmented();
	ADMMAugmented(double* _mu, double* _z, double _rho, double* _w_init);

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
	double* z;
	double rho;
	double* w_init;
	
};

#endif
