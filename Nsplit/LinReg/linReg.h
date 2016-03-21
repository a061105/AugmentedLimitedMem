#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "Matrix.h"
#include "MatrixMath.h"
#include "../Subsolver.h"

using namespace std;

class LinRegSubsolver: public Subsolver{
	
	public:
	Matrix X;
	Matrix y;
	int N, D;

	virtual void readData(char* fname);

	virtual int model_dim();
	
	virtual void subSolve(double* u, double* z, double rho, double* model);

	virtual void writeModel(char* fname, double* model);
};
