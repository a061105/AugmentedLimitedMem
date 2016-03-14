#ifndef CRFSPARSE_SUBSOLVER
#define CRFSPARSE_SUBSOLVER

#include "../Subsolver.h"

#include<math.h>
#include<vector>
#include<cstring>
#include<stdlib.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<deque>
#include<forward_list>
#include <ctime>
#include <iomanip>      // std::setprecision
//problem
#include "problem.h"
#include "seq_label.h"
#include "seq_label_sgd.h"
#include "taxonomy_sgd.h"
#include "taxonomy.h"
//optimizer
//#include "lr.h"
#include "proxLBFGS.h"
#include "proxGrad.h"
#include "SGD.h"
#include "optimizer.h"
#include "OWLQN.h"
#include "BCD.h"
//#include "CD.h"

using namespace std;

class CRFsparse:public Subsolver{

	public:
	
	CRFsparse(int argc, char** argv, int D, int L);

	virtual void readData(char* fname);
	
	virtual void save(char* fname);
	virtual void load(char* fname);
	virtual void release();

	virtual int model_dim();

	virtual void subSolve(Augmented* Aug, double* model);
	virtual void writeModel(char* fname, double* model);
	
	private:
	
	Param param;
	char* train_file;
	char* model_file;
	optimizer* opt;
	Problem* prob;
	int raw_d;
	int numLabel;
	
	void exit_with_help();
	void parse_command_line(int argc, char** argv, char*& train_file, char*& model_file);
};

#endif
