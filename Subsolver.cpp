#ifndef SUBSOLVER_CPP
#define SUBSOLVER_CPP

#include <iostream>
#include <string>
#include "util.h"
//#include "LinReg/linReg.h"
#include "Liblinear/train.h"
#include "CRFsparse/CRFsparse.h"
//#include "SMIDAS/smidas.h"
#include <iostream>

using namespace std;

Subsolver* create_subsolver(char* cmd, int D, int L){
	
	Subsolver* solver;

	string cmd_str(cmd);
	vector<string> cmd_tokens = split(cmd_str," ");
	
	int argc;
	char** argv;
	vectToArr(cmd_tokens, argc, argv);
	
	string solverType = cmd_tokens[0];
	
	if(solverType == "liblinear"){
		solver = new LiblinearSubsolver(argc, argv, D);
	}
	else if(solverType == "CRFsparse"){
		solver = new CRFsparse(argc, argv, D, L);
	}
	//else if(solverType == "smidas"){
	//	solver = new SmidasSubsolver(argc, argv, D);
	//}
	else{
		cerr << "unknown solver: " << cmd_tokens[0] << endl;	
		exit(0);
	}
	
	return solver;
}

#endif
