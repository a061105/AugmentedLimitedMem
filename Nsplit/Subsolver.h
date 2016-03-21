#ifndef SUBSOLVER
#define SUBSOLVER

#include <iostream>
#include <vector>

class Augmented;

class Subsolver{
	      
	public:
	
	virtual void readData(char* fileName)=0;
	virtual void load(char* fileName)=0; //load binary-formatted data
	virtual void save(char* fileName)=0; //save to binary-formatted data
	virtual void release()=0; //release data block

	virtual int model_dim()=0;

	/** min_ f_sub(model) + mu' (model-z) + \rho \|model-z\|^2
	 */
	virtual void subSolve( Augmented* Aug, double* model  )=0;
	virtual void writeModel(char* fileName, double* model)=0;
};

extern Subsolver* create_subsolver(char* cmd, int D, int L); //For CRF

class Augmented{
	
	public:
	virtual double fun(double* w)=0;
	virtual void grad(double* w, double* g)=0;
	virtual void Hs(double* w, double* Hs)=0;
	
	//coordinate-wise
	virtual double fun(double w, int index)=0;
	virtual double grad(double w, int index)=0;
	virtual double Hii(double w, int index)=0;

	virtual double* getModel()=0;
	virtual double* getMu_B()=0;
	virtual void getAlpha(double* alpha)=0;
	virtual void setAlpha(double* alpha)=0;
};

#endif
