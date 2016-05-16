#include<math.h>
#include<vector>
#include<cstring>
#include<stdlib.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<deque>
#include <ctime>
#include <iomanip>  

#include "util.h"
#include "problem.h"
#include "seq_label.h"
#include "multiclass.h"
#include "taxonomy.h"
#include "pseudo_seqlabel.h"

#include "optimizer.h"
#include "proxQN.h"
#include "BCD.h"

using namespace std;

Param param;

void exit_with_help(){

    cerr << "Usage: ./train (options) [train_data] (model)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	0 -- Multiclass Classification (info file = label list)" << endl;
    cerr << "	1 -- Sequence labeling (info file = feature template)" << endl;
    cerr << "	2 -- Hierarchical classification (info file = taxonomy tree)" << endl;
    cerr << "	3 -- Pseudolikelihood Sequence Labeling (info file = label list)" << endl;
    cerr << "-i info: (additional info specified in a file)" << endl;
    cerr << "-m max_iter: maximum_outer_iteration (default 100)" << endl;
    cerr << "-l lambda: L1 regularization coefficient (default 1.0)" << endl;
    cerr << "-t theta:  L2-regularization for unigram (default 0.0)" << endl;
    cerr << "-2 theta2: L2-regularization for bigram  (default 0.0)" << endl;
    cerr << "-e epsilon: stop criterion (default 1e-2)"<<endl;
    cerr << "-s solver: (default 0)" << endl;
    cerr << "	0 -- proxQN" << endl;
    cerr << "	1 -- block coordinate proxQN" << endl;
    exit(0);
}

void parse_command_line(int argc, char** argv, char*& train_file){

    int i;
    for(i=1;i<argc;i++){

        if( argv[i][0] != '-' )
            break;
        if( ++i >= argc )
            exit_with_help();

        switch(argv[i-1][1]){

            case 'p': param.problem_type = atoi(argv[i]);
                      break;
            case 'i': param.info_file = argv[i];
                      break;
            case 'm': param.max_iter = atoi(argv[i]);
                      break;
            case 'l': param.lambda = atof(argv[i]);
                      break;
            case 't': param.theta = atof(argv[i]);
                      break;
	    case '2': param.theta2 = atof(argv[i]);
		      break;
            case 'e': param.epsilon = atof(argv[i]);
                      break;
	    case 's': param.solver = atof(argv[i]);
		      break;
            default:
                      cerr << "unknown option: -" << argv[i-1][1] << endl;
		      exit(0);
        }
    }

    if(i>=argc)
        exit_with_help();

    train_file = argv[i];
    i++;

    if( i<argc )
        param.model_file = argv[i];
    else{
	param.model_file = new char[FNAME_LENGTH];
        strcpy(param.model_file,"model");
    }
}


int main(int argc, char** argv){
    char* train_file;
    parse_command_line(argc, argv, train_file);

    srand(time(NULL));
    optimizer* opt;
    switch(param.solver) {
	case 0:
		cerr << "Use proxQN to solve" << endl;
    		opt = new proxQN();
		break;
	case 1:
		cerr << "Use BC-proxQN to solve" << endl;
		opt = new bCD();
		break;
    }
    
    Problem* prob;
    switch(param.problem_type){
        case 0:
            cerr<<"Multiclass classifiction problem."<<endl;
	    prob = new MulticlassProblem(train_file);
	    break;
        /*case 1:
            cerr<<"Sequence labeling problem."<<endl<<endl;
            prob = new SeqLabelProblem(train_file);
            break;
        case 2:
            cerr<<"Hierarchical classification problem."<<endl<<endl;
            prob = new TaxonomyProblem(train_file);     
            break;
        case 3:
            cerr<<"Pseudolikelihood Sequence Labeling problem."<<endl;
	    prob = new PseudoSeqLabelProblem(train_file);
	    break;*/
        case 4:
            cerr<<"sequence alignment not ready"<<endl;
            exit(0);
        case 5:
            cerr<<"sequence parsing not ready"<<endl;
            exit(0);
    }
    cerr << "number of weights: " << prob->d << endl;
    cerr<<"number of samples: "<< prob->N <<endl<<endl;	
    opt->lambda = param.lambda;
    opt->max_iter = param.max_iter;
    opt->epsilon = param.epsilon;
    
    opt->minimize(prob);

    writeModel(param.model_file, prob->w, prob->d, prob->raw_d);

}
