#ifndef LIBLINEAR_SUBSOLVER
#define LIBLINEAR_SUBSOLVER

#include "../Subsolver.h"
#include "linear.h"
#include "../ADMMAug.h"


class LiblinearSubsolver:public Subsolver{
	
	public:
	
	LiblinearSubsolver(int argc, char** argv, int D);

	virtual void readData(char* fname);
	virtual void save(char* fname);
	virtual void load(char* fname);
	virtual void release();

	virtual int model_dim();

	virtual void subSolve(Augmented* Aug, double* model);
	virtual void writeModel(char* fname, double* model);

	private:
	struct feature_node *x_space;
	struct parameter param;
	struct problem prob;
	struct model* _model;
	int flag_cross_validation;
	int nr_fold;
	double bias;
	
	void parse_command_line(int argc, char **argv);
	void read_problem(const char *filename);
};


#endif
