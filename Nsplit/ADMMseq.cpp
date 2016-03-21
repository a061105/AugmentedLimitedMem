#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include "Subsolver.h"
#include "Swapper.h"
#include "util.h"
#include "SampleModel.h"
#include "ADMMAug.h"
#include "DDAug.h"
#include "proxAug.h"

#define THRESHOLD 1

using namespace std;
SampleWeight* sampleweight;
void dump_pr_dr(ostream& out, int K, int D, double** model, double** model_old, double* z, double rho){
	
	double* pr_arr = new double[K];
	double* dr_arr = new double[K];
	for(int i=0;i<K;i++){
		pr_arr[i] = 0.0;
		dr_arr[i] = 0.0;
	}
	
	//compute dual residual
	for(int i=0;i<K;i++){
		for(int j=0;j<D;j++){
			dr_arr[i] += (model[i][j] - z[j])*(model[i][j] - z[j]);
		}
		dr_arr[i] = sqrt(dr_arr[i]);
	}
	
	//compute primal residual
	for(int i=0;i<K;i++){
		for(int j=0;j<D;j++){
			pr_arr[i] += (model[i][j]-model_old[i][j])*(model[i][j]-model_old[i][j]);
		}
		pr_arr[i] = rho*sqrt(pr_arr[i]);
	}

	for(int i=0;i<K;i++){
		out << "(" << dr_arr[i] << "," << pr_arr[i] << "), " ;
	}

	delete[] pr_arr;
	delete[] dr_arr;
}

void admm_std(Swapper* swapper, int K, int D, double*& z, int max_iter=30 ){
	
	double* model = new double[D];
	double* mu = new double[D];
	int iter = 0;
	double dual_residual; // sum_k \|w_k - z\|_2
	double primal_residual;
	double* z_old = new double[D];
	for(int i=0;i<D;i++) {
		z_old[i] = 0.0;
		model[i] = 0.0;
		mu[i] = 0.0;
	}
	
	ADMMAugmented* admmAug = new ADMMAugmented(mu, z_old, K, model);
	admmAug->rho = K;
	
	Subsolver* solver;
	
	time_t time_cur, time_last=0;
	time(&time_last);
	
	char modelDir[1024];
	sprintf(modelDir,"model_dir.std");
	sampleweight = new SampleWeight(modelDir,-1,1000000,swapper->getSolver(0));
	sampleweight->setModel(z,D);
	sampleweight->start();
	while(iter < max_iter){

		for(int i=0;i<D;i++)
			z[i] = 0.0;
		
		//solve primal problem to obtain model_new, given mu, z
		for(int i=0;i<K;i++){
			map<char*,double*> info;
			info.clear();
			info.insert(pair<char*,double*>("model",model));
			info.insert(pair<char*,double*>("mu",mu));
			swapper->load(i, D, info, solver);
			//admmAug->mu = mu;
			//admmAug->z = z_old;
			//admmAug->w_init = model;
			//admmAug->rho = K;
			solver->subSolve(admmAug, model);
			cerr << "." ;
			
			//cumulated for z
			for(int j=0;j<D;j++){
				z[j] += model[j] + mu[j]/admmAug->rho;
			}
			swapper->save(i, D, info, solver);
		}
		cerr << endl;
		
		//obtain z
		for(int i=0;i<D;i++){
			z[i] /= K;
			if( fabs(z[i]) < 1e-4 ){
				z[i] = 0.0;
			}
		}
		
		
		sampleweight->sample();
		
		//update mu
		map<char*,double*> info;
		info.clear();
		info.insert(pair<char*,double*>("model",model));
		info.insert(pair<char*,double*>("mu",mu));
		for(int i=0;i<K;i++){
			
			swapper->load(i, D, info);
			for(int j=0;j<D;j++){
				mu[j] += 1.0*admmAug->rho*(model[j]-z[j]);
			}
			swapper->save(i, D, info);
		}
		
		//compute dual residual
		dual_residual = 0.0;
		for(int i=0;i<K;i++){
			swapper->load(i, D, info);
			for(int j=0;j<D;j++){
				dual_residual += (model[j] - z[j])*(model[j] - z[j]);
			}
		}
		dual_residual = sqrt(dual_residual);

		//compute primal residual
		primal_residual = 0.0;
		for(int j=0;j<D;j++){
			primal_residual += (z[j]-z_old[j])*(z[j]-z_old[j]);
		}
		primal_residual = admmAug->rho*sqrt(primal_residual);
		
		//print info
		time(&time_cur);
		cerr <<  iter << ", diskI/O=" << iter*K << ", p_r=" << primal_residual <<", d_r=" << dual_residual ;
		//if( primal_residual < THRESHOLD && dual_residual < THRESHOLD )
		//	break;
		cerr <<  ", iterTime=" << time_cur-time_last << endl;
		
		//copy z to z_old
		for(int i=0;i<D;i++)
			z_old[i] = z[i];
		iter++;
	}
	
	delete[] z_old;
	delete[] model;
	delete[] mu;
	delete admmAug;
}

void uniform_random(vector<int>& nodelist, int& r, int K){
	
	if( nodelist.size() < K ){
		
		nodelist.clear();
		for(int i=0;i<K;i++)
			nodelist.push_back(i);
		r = 0;
		shuffle(nodelist, K);
		
		return ;
	}

	if( r+1 < K ){
		r ++;
	}
	else{
		shuffle(nodelist, K);
		r = 0;
	}
}

int sum_to(int x) {
	int sum=0;
	for(int i=0;i<x;i++)
		 sum += (i+1);
	return sum;
}

void shrink_uniform_random(vector<int>& nodelist, int& r, int K, int shrink_block){// if shrink_block < 0 => shuffle, otherwise shrink the block
	if(shrink_block >= 0) {
		nodelist.erase(remove(nodelist.begin(), nodelist.end(), shrink_block), nodelist.end());
		return;
	}	
	if( r+1 < nodelist.size() ){
		r ++;
	}
	else{
		nodelist.clear();
		for(int i=0;i<K;i++)
			nodelist.push_back(i);
		shuffle(nodelist, nodelist.size());
		r = 0;
	}
}


// \|w_k-z_t\|^2 = 0 && \|z_{t} - z_old{k}\|^2 = 0 

//admm_RBCD
void admm_RBCD(Swapper* swapper, int K, int D, double*& z, int _max_iter=30 )
{
	int r = 0; //cyclic reader
	
	int max_iter = _max_iter*K;
	int max_inner_iter = 1;
	int iter = 0;
	double primal_residual, dual_residual;
	double ratio = 0.1;
	int next_iter = K-1;

	double* model = new double[D];
	double* mu = new double[D];
	ADMMAugmented* admmAug = new ADMMAugmented(mu, z, K, model);
	double rho_K = admmAug->rho*K;
	double* z_sub = new double[D];
	double** model_old = new double*[K];
	for(int i=0;i<K;i++)
		model_old[i] = new double[D];
	
	for(int i=0;i<D;i++){
		z[i] = 0.0;
	}
	
	time_t time_cur, time_last=0;
	time(&time_last);
	char modelDir[1024];
	sprintf(modelDir,"model_dir.RBCD");
	sampleweight = new SampleWeight(modelDir,-1,1000000,swapper->getSolver(0));
	sampleweight->setModel(z,D);
	sampleweight->start();
	
	Subsolver* solver;
	//initialized models and z[.]
	for(int k=0;k<K;k++){
		map<char*,double*> info;
		info.insert(pair<char*,double*>("model",model));
		info.insert(pair<char*,double*>("mu",mu));
		swapper->load(k, D, info, solver);
		//solve model
		solver->subSolve(admmAug, model);
		cerr << "." ;
		//solve z
		for(int j=0;j<D;j++)
			z[j] += model[j]/K ;
		swapper->save(k, D, info, solver);
		for(int j=0;j<D;j++)
			model_old[k][j] = model[j];
	}
	cerr << endl;
	sampleweight->sample();

	//Main Loop
	vector<int> nodelist;
	nodelist.clear();	
	for(int i=0;i<K;i++)
		nodelist.push_back(i);
	r = 0;
	shuffle(nodelist, K);
	while(iter < max_iter){
		
		//get next random index
		uniform_random(nodelist, r, K);
		//shrink_uniform_random(nodelist, r, K, -1);
		int k = nodelist[r]; 
		cerr << ".";
		if(r == 0){
			cout << "iter:" << iter << endl;
		}
		//get model, mu, solver for the k-th block
		map<char*,double*> info;
		info.insert(pair<char*,double*>("model",model));
		info.insert(pair<char*,double*>("mu",mu));
		swapper->load(k, D, info, solver);
		for(int j=0;j<D;j++)
			model_old[k][j] = model[j];

		int inner_iter = 0;
		while(inner_iter < max_inner_iter)
		{	
			//update mu
			for(int j=0;j<D;j++)
				z_sub[j] = mu[j]/rho_K; //deduct mu_old
			for(int j=0;j<D;j++)
				mu[j] += 1.0*admmAug->rho*(model[j]-z[j]);
			
			//solve primal problem to obtain model_new, given mu, z
			for(int j=0;j<D;j++)
				z_sub[j] += model[j]/K; //deduct model_old
			
			solver->subSolve(admmAug, model);
			
			//solve z
			for(int j=0;j<D;j++){
				z[j] +=  (model[j]+mu[j]/admmAug->rho )/K - z_sub[j];
			}
			
			inner_iter++;
		}
		swapper->save(k, D, info, solver);
		//print info
		if(iter % K == K-1){
		//if(next_iter == iter) {
			for(int j=0;j<D;j++){
				if( fabs(z[j]) < 1e-4 )
					z[j] = 0.0;
			}
			cerr << endl;	
			sampleweight->sample();
			//compute dual residual
			double block_dual_residual[K];
			double max_dual_residual = 0;
			//dual_residual = 0.0;
			cout << "d_r= " ;
			map<char*,double*> info;
			info.insert(pair<char*,double*>("model",model));
			info.insert(pair<char*,double*>("mu",mu));
			for(int i=0;i<K;i++){
				block_dual_residual[i] = 0;
				swapper->load(i, D, info);	
				for(int j=0;j<D;j++)
					block_dual_residual[i] += (model[j] - z[j])*(model[j] - z[j]);
				dual_residual += block_dual_residual[i];
				block_dual_residual[i] = sqrt(block_dual_residual[i]);
				if(max_dual_residual < block_dual_residual[i])
					max_dual_residual = block_dual_residual[i];
				//cout << i << ":" << block_dual_residual[i] << " ";
			}
			cerr << endl;
			dual_residual = sqrt(dual_residual);
			
			//compute primal residual
			double block_primal_residual[K];
			double max_primal_residual = 0;
			primal_residual = 0.0;
			//cout << "p_r=" ;
			for(int i=0;i<K;i++){
				block_primal_residual[i] = 0;
				swapper->load(i, D, info);
				for(int j=0;j<D;j++)
					block_primal_residual[i] += (model[j]-model_old[i][j])*(model[j]-model_old[i][j]);
				primal_residual += block_primal_residual[i];
				block_primal_residual[i] = sqrt(block_primal_residual[i]);
				if(max_primal_residual < block_primal_residual[i])
					max_primal_residual = block_primal_residual[i];
				//cout << i << ":" << block_primal_residual[i] << " ";

			}
			primal_residual = sqrt(primal_residual);
			//cerr << endl;
			/*cout << "size:" <<  nodelist.size() << endl;
			//check which block should be shrinked
			for(int i=0;i<K;i++) {
				if(ratio*max_primal_residual > block_primal_residual[i] && ratio*max_dual_residual > block_dual_residual[i]) {
					//cout << ratio*max_primal_residual << " ,i=" << i << " " << block_primal_residual[i] << endl;
					shrink_uniform_random(nodelist, r, K, i);
				}
			}
			for(int i=0;i<nodelist.size();i++)
				cout << nodelist[i] << " ";
			cout << endl;
			cout << "# of left blocks:" << nodelist.size() << "/" << K << endl;
			primal_residual = rho*sqrt(primal_residual/K);*/
			
			
			//compute time spent on current iteration
			time(&time_cur);
			cout <<  iter << ", diskI/O=" << iter << ", p_r=" << primal_residual << ", d_r=" << dual_residual
			     <<  ", iterTime=" << time_cur-time_last << endl;

			//shrink iter
			//next_iter = iter + nodelist.size();
			//cout << "iter=" << iter << ",next iter=" << next_iter << endl;
		}
		//if( primal_residual < THRESHOLD && dual_residual < THRESHOLD )
		//	break;
		
		
		iter++;
	}

	delete[] z_sub;
	delete[] model;
	delete[] mu;
}

void DualDecomposion(Swapper* swapper, int K, int D, int max_iter=30 ){
	
	int iter = 0;
	double dual_residual; // sum_k \|w_k - z\|_2
	double primal_residual;
	double* model = new double[D];
	DDAugmented* ddAug = new DDAugmented(model);
	ddAug->mu = new double[D];// mu not in block
	Subsolver* solver;
	int N = swapper->N;
	cout << N << endl;
	int block_size = N/K;
	ddAug->global_alpha = new double[N];

	for(int i=0;i<D;i++) {
		ddAug->mu[i] = 0.0;
		model[i] = 0.0;
	}
	for(int i=0;i<N;i++)
		ddAug->global_alpha[i] = 0.0;
	
	time_t time_cur, time_last=0;
	time(&time_last);
	
	char modelDir[1024];
	sprintf(modelDir,"model_dir.dd");
	sampleweight = new SampleWeight(modelDir,-1,1000000,swapper->getSolver(0));
	sampleweight->setModel(model,D);
	sampleweight->start();
	while(iter < max_iter){

		//solve primal problem to obtain model_new, given mu
		for(int i=0;i<K;i++){
			map<char*,double*> info;
			info.clear();
			swapper->load(i, D, info, solver);
			ddAug->instance.clear();
			if(i == K-1) {
				for(int j=0;j<block_size+(N%K);j++)
					ddAug->instance.push_back(i*block_size+j);
			}else {
				for(int j=0;j<block_size;j++) {
					ddAug->instance.push_back(i*block_size+j);
				}
			}
			solver->subSolve(ddAug, model);
			cerr << "." ;
			swapper->save(i, D, info, solver);
		}
		cerr << endl;
		sampleweight->sample();
		
		cerr << "iter:" << iter << endl;
		iter++;
	}
	
	delete[] model;
}

void proxDBCD_primal(Swapper* swapper, int K, int D, int max_iter) {

	int N = swapper->N;
	int numBlock = K;
	int block_size = N/numBlock;
	vector<int> block_index;
	proxAugmented* proxAug = new proxAugmented();
	
	//initialize w, wt, mu, alpha
	Subsolver* solver;
	proxAug->global_alpha = new double[N];
	for(int i=0;i<N;i++)
		proxAug->global_alpha[i] = 0.0;
	proxAug->mu = new double[D];
	double* w = new double[D];
	proxAug->w_init = w;
	double* mu_Bbar = new double[D];
	proxAug->wt = new double[D];
	for(int j=0;j<D;j++){
		w[j] = 0.0;
		proxAug->mu[j] = 0.0;
		proxAug->wt[j] = 0.0;
	}

	char modelDir[1024];	
	sprintf(modelDir,"model_dir.prox");
	sampleweight = new SampleWeight(modelDir,-1,1000000,swapper->getSolver(0));
	sampleweight->setModel(w,D);
	sampleweight->start();

	proxAug->eta_t = 1;
	int iter = 1;
	while(iter <= max_iter){
		for(int i=0;i<K;i++) {
			map<char*,double*> info;
			info.clear();
			swapper->load(i, D, info, solver);
			proxAug->instance.clear();
			if(i == K-1) {
				for(int j=0;j<block_size+(N%K);j++)
					proxAug->instance.push_back(i*block_size+j);
			}else {
				for(int j=0;j<block_size;j++) {
					proxAug->instance.push_back(i*block_size+j);
				}
			}
			solver->subSolve(proxAug, w);
			cerr << "." ;
			swapper->save(i, D, info, solver);
		}
		
		cerr << endl;
		sampleweight->sample();
		if(iter % 1 == 0) {
			for(int j=0;j<D;j++)
				proxAug->wt[j] = w[j];
		}
		cerr << "iter:" << iter << endl;
		iter++;
		
	}
	
	delete[] w;
	return;
}

void online(Swapper* swapper, int K, int D, int max_iter) {

	double* model = new double[D];
	double* mu = new double[D];
	int iter = 0;
	double dual_residual; // sum_k \|w_k - z\|_2
	double primal_residual;
	double* z_old = new double[D];
	for(int i=0;i<D;i++) {
		z_old[i] = 0.0;
		model[i] = 0.0;
		mu[i] = 0.0;
	}
	
	ADMMAugmented* admmAug = new ADMMAugmented(mu, z_old, K, model);
	admmAug->rho = K;
	
	Subsolver* solver;
	
	time_t time_cur, time_last=0;
	time(&time_last);
	
	char modelDir[1024];
	sprintf(modelDir,"model_dir.online");
	sampleweight = new SampleWeight(modelDir,-1,1000000,swapper->getSolver(0));
	sampleweight->setModel(model,D);
	sampleweight->start();
	while(iter < max_iter){

		
		//solve primal problem to obtain model_new, given mu, z
		for(int i=0;i<K;i++){
			map<char*,double*> info;
			info.clear();
			swapper->load(i, D, info, solver);
			solver->subSolve(admmAug, model);
			cerr << "." ;
			
			swapper->save(i, D, info, solver);
		}
		cerr << endl;
		sampleweight->sample();
		
		
		iter++;
	}
	
	delete[] z_old;
	delete[] model;
	delete[] mu;
	delete admmAug;
	return;
}

int main(int argc, char** argv){
	
	if( argc < 4 ){
		cerr << "./ADMMseq [data_dir] [sub_solver] [0:std/1:RBCD/2:Stream/3.DD/4.prox/5.online(-s 14)] (max_iter)" << endl;
		cerr << "Subsolver:" << endl;
		cerr << "'linReg'" << endl;
		cerr << "'liblinear'" << endl;
		exit(0);
	}
	
	srand(time(NULL));
	//srand(1);
	char* data_dir = argv[1];
	char* cmd = argv[2];
	int method = atoi(argv[3]);
	char* modelFile = "model";
	int max_iter;
	if( argc > 4 ){
		max_iter = atoi(argv[4]);
	}else{
		max_iter = 5;
	}
	
	//Setup Swapper from data directory
	Swapper* swapper = new Swapper(data_dir, cmd);
	int D = swapper->model_dim();
	int K = swapper->num_block();

	// model average
	double* z = new double[D];
	for(int i=0;i<D;i++)
		z[i] = 0.0;	
	cerr << "solve" << endl;
	//Main Loop
	if(method==0)admm_std(swapper, K, D,  z, max_iter);
	else if(method==1)admm_RBCD(swapper, K, D, z, max_iter);
	//else if(method==2)admm_Stream(swapper, K, D, z);
	else if(method==3)DualDecomposion(swapper, K, D, max_iter);
	else if(method==4)proxDBCD_primal(swapper, K, D, max_iter);
	else if(method==5)online(swapper, K, D, max_iter);
	else{
		cerr << "no such method: " << method << endl;
		exit(0);
	}
	
	// write model
	//swapper->getSolver(0)->writeModel(modelFile, z);
	int nnz=0;
	for(int i=0;i<D;i++)
		if( fabs(z[i]) > 1e-12 )
			nnz++;
	cerr << "nnz=" << nnz << endl;

	delete swapper;

	return 0;
}
