#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
#include "train.h"
#include "../ADMMAug.h"

using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"	14 -- Online L1-regularized logistic regression (primal)\n"
	"	15 -- L1-regularized Regression (primal)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}



LiblinearSubsolver::LiblinearSubsolver(int argc, char **argv, int D)
{
	parse_command_line(argc, argv);
	prob.n = D;

	prob.bias=bias;
	prob.y = NULL;
	prob.x = NULL;
	x_space = NULL;
	//for ADMM
	/*prob.alpha_init = NULL;
	prob.w_init = NULL;*/
}

void LiblinearSubsolver::readData(char* input_file_name){
	
	read_problem(input_file_name);
	const char *error_msg;
	error_msg = check_parameter(&prob,&param);
	
	//force 1/-1 to be the positive/negative label
	/*for(int i=0;i<prob.l;i++){
		if( fabs(prob.y[i]-1.0) < 1e-6){
			prob.y[i] = 1;
		}
		else{
			prob.y[i] = -1;
		}
	}*/
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		(1);
	}
}

void LiblinearSubsolver::subSolve(Augmented* Aug, double* model){
	
	double* mu_B = Aug->getMu_B();
	double* alpha = new double[prob.l];
	if(mu_B != NULL) {// dual decomposition
		Aug->getAlpha(alpha);
		for(int i=0;i<prob.l;i++) {
			feature_node* s = prob.x[i];
			while(s->index != -1) {
				mu_B[s->index-1] -= s->value*alpha[i];
				s++;
			}	
		}
	}
	prob.augmented = Aug;
	/* ADMM
	prob.mu = mu;
	prob.z = z;
	prob.rho = rho;
	prob.w_init = model;*/
	
	_model = train(&prob, &param);
	for(int i=0;i<prob.n;i++){
		model[i] = _model->w[i];
	}
	if(mu_B != NULL) {// dual decomposition
		Aug->getAlpha(alpha);
		// new mu
		for(int i=0;i<prob.l;i++) {
			feature_node* s = prob.x[i];
			while(s->index != -1) {
				mu_B[s->index-1] += s->value*alpha[i];
				s++;
			}	
		}

	}
	
	delete[] alpha;
}

void LiblinearSubsolver::writeModel(char* fname, double* model){
	
	//parse
	_model->w = model;
	
	save_model(fname, _model);
}


void LiblinearSubsolver::parse_command_line(int argc, char **argv)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.max_iter = 1000;
	flag_cross_validation = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'h':
				exit_with_help();
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'M':
				param.max_iter = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);


	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
			case ONLINE_L1LOGI:
				param.eps = 0.1;
				break;
			case LASSO:
				param.eps = 0.1;
		}
	}
}

// read in a problem (in libsvm format)
void LiblinearSubsolver::read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	prob.l = 0;
	elements = 0;
	max_line_len = 1000000;
	line = Malloc(char,max_line_len);
	
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);
	char tmp[1000000];
	fgets(tmp,1000000,fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);
	
	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;
	
	fclose(fp);
}

/** binary format
 *  N D
 *  [0/1] alpha1 ... alphaN
 *  y1 y2 ... yN
 *  #feature_node f1 f2 ....
 */

void LiblinearSubsolver::save(char* fname){
	
	ofstream fout(fname,ios::out|ios::binary);
	//N D
	fout.write((char*)&(prob.l),sizeof(int));
	fout.write((char*)&(prob.n),sizeof(int));
	//alpha
	int tmp =0;
	fout.write((char*)&tmp,sizeof(int));
	//label array
	fout.write((char*)prob.y, sizeof(double)*prob.l);
	//count #feature_node
	int num_node = 0;
	for(int i=0;i<prob.l;i++){
		feature_node* s = prob.x[i];
		while( s->index != -1 )
		{
			num_node++;
			s++;
		}
	}
	//write feature_node array
	fout.write( (char*)&num_node, sizeof(int) );
	fout.write( (char*)x_space, sizeof(struct feature_node)*(num_node+prob.l) );

	fout.close();
}

void LiblinearSubsolver::load(char* fname){
	
	ifstream fin(fname,ios::in|ios::binary);
	//N D
	fin.read((char*)&(prob.l),sizeof(int));
	fin.read((char*)&(prob.n),sizeof(int));// already got from constructor
	
	//alpha
	int tmp =0;
	fin.read((char*)&tmp,sizeof(int));
	//label array
	if(prob.y==NULL)
		prob.y = new double[prob.l];
	fin.read((char*)prob.y, sizeof(double)*prob.l);
	//read feature_node array
	int num_node;
	fin.read( (char*)&num_node, sizeof(int) );
	if(prob.x==NULL){
		prob.x = new feature_node*[num_node];
		x_space = new feature_node[num_node+prob.l];
	}
	fin.read( (char*)x_space, sizeof(struct feature_node)*(num_node+prob.l) );
	//record start of each instance
	feature_node* s = x_space;
	for(int i=0;i<prob.l;i++){
		prob.x[i] = s;
		while( s->index != -1 ) s++;
		s++;
	}

	fin.close();
}
int LiblinearSubsolver::model_dim(){
	return prob.n;
}
void LiblinearSubsolver::release(){
	
	delete[] prob.y;
	prob.y = NULL;
	delete[] prob.x;
	prob.x = NULL;
	delete[] x_space;
	x_space = NULL;
}
