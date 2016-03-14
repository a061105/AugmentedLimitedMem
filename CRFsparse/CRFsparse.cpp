#include "CRFsparse.h"

void CRFsparse::exit_with_help(){

	cerr << "Usage: ./crf-Prox-QN (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-p problem_type: (default 2)" << endl;
	//    cerr << "	0 -- multiclass classification" << endl;
	cerr << "	1 -- classification w/ taxonomy" << endl;
	cerr << "	2 -- sequence labelling (info=feature template)" << endl;
	//    cerr << "	3 -- sequence alignment" << endl;
	//    cerr << "	4 -- sequence parsing" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Proximal Gradient" << endl;
	cerr << "	1 -- Proximal Quasi-Newton" << endl;
	cerr << "	2 -- OWL-QN" << endl;
	cerr << "	3 -- Stochastic Gradient Descent" <<endl;
	cerr << "	4 -- Block Coordinate Descent" <<endl;
	//    cerr << "	5 -- coordinate descent" <<endl;
	cerr << "-i info: (additional info specified in a file)" << endl;
	cerr << "-m max_iter: maximum_outer_iteration" << endl;
	cerr << "-l lambda: regularization coeeficient (default 1.0)"<<endl;	
	cerr << "-e epsilon: stop criterion (default 1e-6)"<<endl;
	cerr << "-t eta0: initial learning rate for SGD (default 1e-2)"<<endl;
	//    cerr << "-f alpha: exponential decay base for SGD (default 0.9)"<<endl;
	exit(0);
}

void CRFsparse::parse_command_line(int argc, char** argv, char*& train_file, char*& model_file){

	int i;
	for(i=1;i<argc;i++){

		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){

			case 'p': param.problem_type = atoi(argv[i]);
				  break;
			case 's': param.solver = atoi(argv[i]);
				  break;
			case 'i': param.info_file = argv[i];
				  break;
			case 'm': param.max_iter = atoi(argv[i]);
				  break;
			case 'l': param.lambda = atof(argv[i]);
				  break;
			case 'e': param.epsilon = atof(argv[i]);
				  break;
			case 't': param.eta0 = atof(argv[i]);
				  break;
			case 'f': param.alpha = atof(argv[i]);
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
		}
	}

	//if(i>=argc)
	//	exit_with_help();
	if(i<argc){
		train_file = argv[i];
		i++;
	}
	/*if( i<argc ){
		model_file = new char[FNAME_LENGTH];
		model_file = argv[i];
	}else{

		strcpy(model_file,"model");
	}*/
}


CRFsparse::CRFsparse(int argc, char** argv, int D, int L){

	
	parse_command_line(argc, argv, train_file, model_file);
	
	srand(time(NULL));

	switch(param.solver){
		case 0:
			opt = new proxGrad();
			break; 
		case 1:
			opt = new proxLBFGS();
			break;
		case 2:
			opt = new OWLQN();
			break;
		case 4:
			opt = new BCD();
			break;
		default:
			cerr << "solver " << param.solver << " is not supported." << endl;
			exit(0);
	} 
	opt->lambda = param.lambda;
	opt->max_iter = param.max_iter;
	opt->epsilon = param.epsilon;
	prob = new SeqLabelProblem(D,L);
}


void CRFsparse::readData(char* fname){
	
	prob->readProblem(fname, param.info_file);
}

void CRFsparse::save(char* fname){
	
	prob->save(fname);
}

void CRFsparse::load(char* fname){
	
	prob->load(fname);
}

void CRFsparse::release(){
	
	prob->release();
}	
int CRFsparse::model_dim(){
	return prob->d;
}


void CRFsparse::subSolve(Augmented* Aug, double* model){
	prob->augmented = Aug;
	prob->w = prob->augmented->getModel();
	/*
	prob->u = u;
	prob->z = z;
	prob->w = model;
	prob->rho = rho;*/
	
	opt->minimize(prob);
}

void CRFsparse::writeModel(char* fname, double* model){
	
}
