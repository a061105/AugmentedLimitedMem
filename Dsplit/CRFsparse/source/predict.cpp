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
//problem
#include "problem.h"
#include "seq_label.h"
#include "taxonomy.h"
#include "multiclass.h"
//optimizer
#include "optimizer.h"
#include "proxQN.h"
using namespace std;

Param param;

void exit_with_help(){

    cerr << "Usage: ./predict (options) [test_data] [model_file] (prediction_result)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	0 -- Multiclass classification (info file = label list)" << endl;
    cerr << "	1 -- Sequence labeling (info file = feature template)" << endl;
    cerr << "	2 -- Hierarchical classification (info file = taxonomy tree)" << endl;
//    cerr << "	3 -- sequence alignment" << endl;
//    cerr << "	4 -- sequence parsing" << endl;
    cerr << "-i info: (additional info specified in a file)" << endl;
    cerr << "-m method: 0:viterbi, 1:marginal, >1 : marginal w/ 'eta=m' (default 1)" << endl;
    exit(0);
}

void parse_command_line(int argc, char** argv, char*& test_file, char*& model_file, char*& output_file){

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
            case 'm': param.predict_method = atoi(argv[i]);
                      break;
            default:
                      cerr << "unknown option: -" << argv[i-1][1] << endl;
        }
    }

    if(i>=argc-1)
        exit_with_help();

    test_file = argv[i++];

    model_file = argv[i++];

    if (i<argc)
	    output_file = argv[i];
    else strcpy(output_file,"prediction_result");
}

int main(int argc, char** argv){
    char* test_file;
    char* model_file;
    char* output_file = new char[FNAME_LENGTH];
    parse_command_line(argc, argv, test_file, model_file, output_file);
    srand(time(NULL));
    Problem* prob;
    switch(param.problem_type){
        case 0:
            cerr<<"Multiclass classifiction."<<endl;
	    prob = new MulticlassProblem(model_file, test_file);
	    break;
        case 1:
            cerr<<"Sequence labeling"<<endl;
            prob = new SeqLabelProblem(model_file, test_file);
            break;
       case 2:
            cerr<<"Hierarchical classification."<<endl;
            prob = new TaxonomyProblem(model_file, test_file);     
            break;
       case 3:
            cerr<<"sequence alignment not ready"<<endl;
            exit(0);
        case 4:
            cerr<<"sequence parsing not ready"<<endl;
            exit(0);
    }
    cerr<<"number of testing samples: "<< prob->N<<endl;	
    prob->test_accuracy(output_file);
}

