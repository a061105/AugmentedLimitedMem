#include "proxQN.h"
#include "util.h"
#include<math.h>
#include<vector>
#include<stdlib.h>
#include<iomanip>
#include<iostream>
#include<deque>
#include <ctime>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef deque< deque<double> > qmat;

void proxQN::minimize(Problem* prob){

	//----------problem parameters and initial values---------- 
	Int n = prob->N; //sample size
	Int db=prob->d;  //dimension
	ValueMap& w = prob->w; //weights
	w = new double[db];
	for(Int i=0;i<db;i++)
		w[i] = 0.0;
	
	Int i;
	//for(i=0;i<db;i++)
	//	w[i] = 0.0; 	
	
	ValueMap g = createValueMap(db);  //gradient
	ValueMap newg = createValueMap(db); //new gradient
	ValueMap w_change = createValueMap(db); //descent direction

	for (i=0;i<db;i++){
		w_change[i] = 0.0;
	}

	vector<pair<Int,double> > fv_change; //factor value change
	vector<double> shg;//shrinked g: only store partial gradient corresponding to active set
	double curF; //current objective
	double stopCrit; //stopping criterion

	Int max_inner = 10;//maximal inner iterations
	Int i_iter; //actual inner iterations

	//----------line search parameters---------- 
	double sigma = 0.001; //parameter for Armijo rule
	double alpha; //step size
	double candF; //candidate objective value
	double armijo, delta_part, alpha_old; //some temporary variables

	//----------shrinking parameters---------  
	vector<Int> act_set;   //active set (also called working set)
	vector<Int> oldact;   //active set of previous iteration
	vector<Int> innerAct; //active set in inner loop: a shuffle of active set
	for (i=0;i<db;i++)
		act_set.push_back(i);

	double subgrad; //subgradient 
	double normsg0; //l1 norm of the subgradients for the initial variables, 
	double normsg;  //l1 norm of the subgradients 
	double M=0;
	double Mout=0.01*lambda*n; //the tolerance for a variable to be active. After initialization, it is the infinity norm of the subgradients. 
	Int epoch_iter = -1; // iteration number in each epoch

	double epsilon_epoch = epsilon; //stopping epsilon for every epoch in the shrinking
	if (epsilon_epoch<0.0001)
		epsilon_epoch = 0.0001;

	//----------LBFGS parameters----------
	Int m=10; //LBFGS memory size upper bound
	double gamma = 1.0; //The initial Hessian approximation B_0 = gamma I
	Int memo_size, cur_dbl_memo_size, new_dbl_memo_size; //current memory size, 2*memory_size and 2*new_memory_size
	//following notations are almost same as those in paper
	ValueMap Bdiag = createValueMap(db); //diagonal elements of B 
	for (i=0; i<db;i++)
		Bdiag[i] = 0.0;

	ValueMap Bd = createValueMap(db);    //B*w_change
	ValueMap s = createValueMap(db);
	ValueMap y = createValueMap(db);
	deque<double> Ddiag,dhat;
	qmat L,Sps;

	ArrMap Q = createArrMap(db);
	ArrMap Qhat = createArrMap(db);
	for (i=0;i<db;i++){
		Q[i] = new double[2*m];
		Qhat[i] = new double[2*m];
	}
	
	ValueMap* S = new ValueMap[m];
	ValueMap* Y = new ValueMap[m];
	for (i=0;i<m;i++){
		Y[i] = createValueMap(db);
		S[i] = createValueMap(db);
	}

	//----------temporary variables----------
	double l2_reg, nonsmooth, absg, c, z, Spsij, Lij, newdiag, RQfactor;  
	size_t actInd; //index in active set
	Int j,k,qm;
	vector<Int>::iterator ii;

	cerr << "Optimizer: Proximal Quasi-Newton"<<endl;

	cerr <<setw(6)<<"iter"<<setw(20)<<"time(s)"<<setw(20)<<"objective"<< setw(12)<<"nnz"<<setw(20)<<"train accuracy"<< setw(20) << "l1-norm" <<endl;
	clock_t start_time = clock();
	
	pair<Int,Int> range;
	range = make_pair(0,db);
	//initial gradient and objective 
	prob->grad(act_set,range,shg);
	for (i=0;i<db;i++){
		//g[i] = shg[i] + theta*w[i];
		g[i] = shg[i];
	}
	nonsmooth = lambda * l1_norm(w,act_set);
	//l2_reg = theta * l2_norm_sq(w,act_set)/2.0;
	//curF = prob->fun() + nonsmooth + l2_reg;
	curF = prob->fun() + nonsmooth;
	
	clock_t initial_time = clock();
	cerr<< std::setprecision(15);
	/*cerr <<setw(6)<<"0"<<setw(20)<<(double)(initial_time - start_time) / CLOCKS_PER_SEC * 1.0<<setw(20)<<curF<< setw(12)<<act_set.size()<<setw(20)<<prob->train_accuracy()<< setw(20) << nonsmooth << endl;*/

	//***************** outer loop starts ********************

	for (Int iter = 0; iter<max_iter; iter++){
		M = 0;
		epoch_iter++;
		if (epoch_iter<m){
			memo_size = epoch_iter+1;
			cur_dbl_memo_size =2* epoch_iter; //size of dhat, Q and Qhat
		}
		else{
			memo_size = m;
			cur_dbl_memo_size =2* m;
		}

		new_dbl_memo_size = 2*memo_size;
		dhat.clear();
		for (j = 0; j < cur_dbl_memo_size; j++)
			dhat.push_back(0.0);

		for (ii = act_set.begin(); ii != act_set.end(); ++ii)
			w_change[*ii] = 0.0;

		//********** find active set **********

		oldact = act_set;
		act_set.clear();

		normsg = 0.0;

		for (vector<Int>::iterator jj=oldact.begin();jj!=oldact.end();++jj){
			j = *jj;
			if (w[j] <  -1.0e-20){
				subgrad = g[j] - lambda;
				act_set.push_back(j);
				normsg += fabs(subgrad); 
				M=fmax(M,fabs(subgrad));
			}
			else if (w[j]> 1.0e-20){
				subgrad = g[j] + lambda;
				act_set.push_back(j);
				normsg += fabs(subgrad); 
				M=fmax(M,fabs(subgrad));
			}
			else{
				absg = fabs(g[j])-lambda;
				subgrad = fmax(absg,0.0);
				if (absg > -Mout/n){
					act_set.push_back(j);
					normsg += subgrad; 
					M=fmax(M,subgrad);
				}
			}
		}


		//************ termination and new epoch critiion ************

		if (iter == 0)
			normsg0 = normsg;
		stopCrit = normsg/normsg0;

		if (stopCrit < epsilon_epoch){
			if (oldact.size() == db && stopCrit < epsilon ){
				cerr << "termination criterion attained, iter: "<< iter<<endl;
				break;
			}
			else {
				cerr<<"******************** new epoch start ******************"<<endl;
				// when a new epoch starts, the active set is the whole set and the momery of LBFGS is cleared. 
				iter--;
				act_set.clear();
				for (j=0;j<db;j++)
					act_set.push_back(j);
				Mout = 0.01 * lambda * n;
				
				Ddiag.clear(); L.clear(); Sps.clear();
				for (j=0;j<db;j++){
					Bdiag[j]=0.0;
					Bd[j]=0.0;
				}

				prob->grad(act_set,range,shg);
				for (j=0;j<db;j++)
					//g[j] = shg[j] + theta*w[j];
					g[j] = shg[j];

				epoch_iter = -1;

				epsilon_epoch = epsilon_epoch/10.0; 
				if (epsilon_epoch<epsilon)
					epsilon_epoch = epsilon;
				
				continue;
			}
		}
		Mout = M;

		for (ii = act_set.begin(); ii != act_set.end(); ++ii){
			Bdiag[*ii] = gamma;
			for(k=0; k< cur_dbl_memo_size; k++)
				Bdiag[*ii] -= Q[*ii][k]*Qhat[*ii][k];
		}

		innerAct = act_set; 
		i_iter = (Int)db/act_set.size();
		if (i_iter>max_inner)
			i_iter = max_inner;

		//*********** inner loop **************
		for (j = 0; j< i_iter; j++){
			shuffle(innerAct);
			for (ii = innerAct.begin(); ii != innerAct.end(); ++ii){ 
				actInd = *ii;
				Bd[actInd] = gamma*w_change[actInd];

				for (k = 0; k < cur_dbl_memo_size; k++)
					Bd[actInd] -= Q[actInd][k]*dhat[k];

				c = w[actInd]+w_change[actInd];
				z = -c+softThd(c-(g[actInd]+Bd[actInd])/Bdiag[actInd],lambda/Bdiag[actInd]);
				w_change[actInd] += z;

				for (k = 0; k < cur_dbl_memo_size; k++)
					dhat[k] += z*Qhat[actInd][k];
			}
		}

		//************ line search ************
		alpha = 1.0;


		delta_part = - lambda * l1_norm(w,act_set);
		for (ii = act_set.begin();ii != act_set.end();++ii){
			j = *ii;
			delta_part += w_change[j]*g[j];
		}
		prob->compute_fv_change(w_change,act_set,range,fv_change);

		prob->update_fvalue(fv_change,alpha);

		for (ii=act_set.begin(); ii != act_set.end(); ++ii){
			j=*ii;
			w[j] += alpha * w_change[j];
		}

		nonsmooth = lambda * l1_norm(w,act_set); 
		//l2_reg = theta * l2_norm_sq(w,act_set)/2.0;
		//candF = prob->fun() + nonsmooth + l2_reg;
		candF = prob->fun() + nonsmooth ;
		
		armijo = sigma * (nonsmooth + delta_part);

		while ( !(candF <= curF + alpha * armijo) ){
			alpha_old = alpha;
			alpha /= 2.0;
			if (alpha<1e-20){
				cerr<<"exit: step size alpha is too small."<<endl<<"try a greater epsilon."<<endl;
				exit(0);
			}
			prob->update_fvalue(fv_change, alpha - alpha_old);

			for (ii = act_set.begin(); ii != act_set.end(); ++ii){
				k=*ii;
				w[k] += (alpha - alpha_old) * w_change[k];
			}
			nonsmooth =  lambda*l1_norm(w,act_set);
			//l2_reg = theta*l2_norm_sq(w,act_set)/2.0;
			//candF = prob->fun() + nonsmooth + l2_reg;
			candF = prob->fun() + nonsmooth;
		}
		curF = candF;
		
		//********** LBFGS update **************
		// update w,g,s,y,gamma

		prob->grad(act_set,range,shg);
		
		for (i=0;i<act_set.size();i++){ 
			j = act_set[i];
			y[j] = shg[i] - g[j];
			s[j] = alpha*w_change[j];
			//g[j] = shg[i] + theta*w[j];
			g[j] = shg[i];
		}
		
		newdiag = 0; 
		gamma = 0;  
		for (ii = act_set.begin(); ii != act_set.end(); ++ii){
			j = *ii;
			newdiag += s[j] * y[j];
			gamma += s[j] * s[j];
		} 
		gamma = newdiag/gamma; 
		if (newdiag<0)
			cerr<<"s*y< 0 alert,at iter="<<iter<<endl;

		//update L
		if (epoch_iter < m){
			Sps.push_back(deque<double>());
			L.push_back(deque<double>());
			for (j=0;j<epoch_iter;j++){
				Lij = 0.0;
				Spsij = 0.0;
				for (ii = act_set.begin(); ii != act_set.end(); ++ii){
					k = *ii;
					Lij +=s[k]*Y[j][k];
					Spsij += s[k]*S[j][k];
				}
				L[j].push_back(0.0);
				L.back().push_back(Lij);
				Sps[j].push_back(Spsij);
				Sps.back().push_back(Spsij);
			}
			L.back().push_back(0.0);
			Sps.back().push_back(newdiag/gamma);
		}
		else {
			L.pop_front();
			Sps.pop_front();
			for (j=0;j<m-1;j++){
				L[j].pop_front();
				L[j].push_back(0.0);
				Sps[j].pop_front();
			}
			L.push_back(deque<double>());
			Sps.push_back(deque<double>());
			for (j=1;j<m;j++){
				qm = (j+epoch_iter) % m;
				Lij = 0.0;
				Spsij = 0.0;
				for (ii = act_set.begin(); ii != act_set.end(); ++ii){
					k = *ii;
					Lij += s[k]*Y[qm][k];
					Spsij += s[k]*S[qm][k];
				}   
				L.back().push_back(Lij);
				Sps[j-1].push_back(Spsij);
				Sps.back().push_back(Spsij);
			}
			L.back().push_back(0.0);
			Sps.back().push_back(newdiag/gamma);
		}

		//update S, Y and Ddiag
		if (epoch_iter < m){
			for (ii = act_set.begin(); ii != act_set.end(); ++ii){
				j = *ii;
				S[epoch_iter][j] = s[j];
				Y[epoch_iter][j] = y[j];
			}
		}
		if (epoch_iter >= m){
			Ddiag.pop_front();
			qm = epoch_iter % m;
			for (ii = act_set.begin(); ii != act_set.end(); ++ii){
				j = *ii;
				S[qm][j] = s[j];
				Y[qm][j] = y[j];
			}
		}

		Ddiag.push_back(newdiag);

		//update Q
		for (ii = act_set.begin(); ii != act_set.end(); ++ii){
			j = *ii;

			if (epoch_iter<m){
				for (k=0;k<memo_size;k++){
					Q[j][k] = gamma*S[k][j];
					Q[j][k+memo_size] = Y[k][j];
				}
			}
			else{
				for (k=0;k<memo_size;k++){
					qm  = (epoch_iter+1+k) % m;
					Q[j][k] = gamma*S[qm][j];
					Q[j][k+memo_size] = Y[qm][j];
				}
			}
		}

		//compute R and update Qhat
		MatrixXd invr=MatrixXd::Zero(new_dbl_memo_size,new_dbl_memo_size);
		for (i=0;i<memo_size;i++ ){
			for (Int j=0;j<memo_size;j++){
				invr(i,j) = gamma * Sps[i][j];
			}
		}
		MatrixXd matL(memo_size,memo_size);
		for (i=0;i<memo_size;i++ )
			for (Int j=0;j<memo_size;j++)
				matL(i,j) = L[i][j];
		for (i=0;i<memo_size;i++)
			invr(i+memo_size,i+memo_size) = - Ddiag[i];
		invr.topRightCorner(memo_size,memo_size) = matL;
		invr.bottomLeftCorner(memo_size,memo_size) = matL.transpose();
		MatrixXd R(new_dbl_memo_size,new_dbl_memo_size);

		R = invr.inverse();

		for (ii = act_set.begin(); ii != act_set.end(); ++ii){
			i = *ii;
			double* qhat;
			if( (qhat=Qhat[i]) == NULL )
			       Qhat[i] = new double[m];
			for (Int j=0;j<new_dbl_memo_size;j++){
				RQfactor=0.0;
				for (Int k=0;k<new_dbl_memo_size;k++)
					RQfactor += Q[i][k]*R(k,j);
				Qhat[i][j] = RQfactor;
			}
		}

		clock_t cur_time = clock();
		cerr <<setw(6) << iter+1 <<setw(20)<< (double)(cur_time - start_time)/CLOCKS_PER_SEC * 1.0 <<setw(20)<<curF<<setw(12)<<act_set.size() << setw(20)<<prob->train_accuracy()<< setw(20) << nonsmooth << endl;
		
		writeModel(param.model_file, w, db, prob->raw_d);
	}

	delete [] g;
	delete [] newg;
	for (i=0;i<m;i++)
		delete [] S[i];
	delete [] S; 
	delete [] s;
	for (i=0;i<m;i++)
		delete [] Y[i];
	delete [] Y;
	delete [] y;
	delete [] Bdiag;
	delete [] Bd;
	delete [] w_change;
}
