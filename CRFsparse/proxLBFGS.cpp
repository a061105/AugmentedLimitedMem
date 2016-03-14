#include "proxLBFGS.h"
#include<math.h>
#include<vector>
#include<stdlib.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<deque>
#include<forward_list>
#include <ctime>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef deque< deque<double> > qmat;
double l1(double *w, vector<int> &actset){
	double l1norm =  0;
	for (vector<int>::iterator ii=actset.begin();ii != actset.end();++ii)
		l1norm += fabs(w[*ii]);
	return l1norm;
}

double l2(double *w, vector<int> &actset){
	double l2norm =  0;
	for (vector<int>::iterator ii=actset.begin();ii != actset.end();++ii)
		l2norm += w[*ii]*w[*ii];
	return l2norm;
}

double dot(double* a, double* b, vector<int>& actset){

	double sum =0.0;
	for (vector<int>::iterator ii=actset.begin();ii != actset.end();++ii)
		sum += a[*ii]*b[*ii];
	return sum;
}

proxLBFGS::proxLBFGS(){};

void proxLBFGS::minimize(Problem* prob){
	//cerr<<"maxiter:"<<max_iter<<endl;
	//cerr<<"lambda "<<lambda<<endl;
	int m=10;
	double* w = prob->w;
	/*double* u_admm = prob->u;  //ADMM
	  double* z_admm = prob->z;
	  double rho = prob->rho;*/


	//compute factor values based on current w
	prob->compute_fvalue();

	int max_inner = 10;
	int i,j,k,qm;
	vector<int>::iterator ii;
	double sigma = 0.001;
	//double epsilon = 1e-8;
	//cerr<<"epsilon "<<epsilon<<endl;
	double epsilon_shrink = 1000.0*epsilon;
	if (epsilon_shrink<0.0001)
		epsilon_shrink = 0.0001;

	double gamma = 1.0;
	double normsg0,normsg;
	int n = prob->N;
	int db=prob->d;

	/*for(int i=0;i<db;i++)
	  w[i] = 0.0; 	*/

	vector<int> actset;
	for (i=0;i<db;i++)
		actset.push_back(i);
	double subgrad;
	double absg;
	double alpha;
	int miter = 0;
	double c,z;   //buf for inner loop
	size_t actInd; //index in active set

	double Mout=0.01*lambda*n;
	int lbfgsm = -1;
	double M=0;

	vector<pair<int,double> > fv_change;
	vector<double> shg;//shrinked g:only store active g;
	double *g = new double[db];
	double *newg = new double[db];
	double *Bdiag = new double[db];
	double *Bd = new double[db];
	double *d = new double[db];
	for (i=0;i<db;i++)
		d[i]=0.0;
	double *wtemp = new double[db];
	for (i=0;i<db;i++){
		wtemp[i] = w[i];
		if(w[i] != 0.0)
			cout << w[i] << "," << i << endl;
		d[i] = 0.0;
	}


	double *s = new double[db];
	double *t = new double[db];

	deque<double> Ddiag,dhat;
	qmat L,Sps;
	double **Q = new double*[db];
	double **Qhat = new double*[db];
	for (i=0;i<db;i++){
		Q[i] = new double[2*m];
		Qhat[i] = new double[2*m];
		/*        for (j = 0;j<2*m;j++){
			  Q[i][j] = 0.0;
			  Qhat[i][j] = 0.0;
			  }
			  */
	}

	double** S = new double*[m];
	double** T = new double*[m];
	for (i=0;i<m;i++){
		T[i] = new double[db];
		S[i] = new double[db];
	}
	cerr<<"Time(ms) Relative-function-value-difference"<<endl;
	cout<<"Iter nnz"<<endl;

	//clock_t touterb = clock();

	prob->grad(actset,shg);
	//for ADMM
	int index;
	for(int i=0;i<actset.size();i++){
		index = actset[i];
		//shg[i] += u_admm[index] + rho*(w[index]-z_admm[index]);
		shg[i] += prob->augmented->grad(w[index],index);
	}
	//double curF = prob->fun()+ lambda * l1(w,actset) + dot(u_admm,w,actset) + rho/2*( l2(w,actset) + dot(w,z_admm,actset) );
	double curF = prob->fun()+ lambda * l1(w,actset);
	for(int i=0;i<actset.size();i++) {
		index = actset[i];
		curF += prob->augmented->fun(w[index],index);
	}
	//l1r_lr_obj(n, db, w, pred,  labels, lambda,actset);
	double candF;

	int matsize,dbmatsize;
	int dblmiter;
	vector<int> oldact;
	vector<int> innerAct;
	//***************** outer loop starts ********************
	//clock_t firsti = clock();
	//    cerr<<std::setprecision(20)<<(double)(firsti - touterb) / CLOCKS_PER_SEC *1000.0<<" ";
	//cerr <<std::setprecision(20)<<(double)(firsti - touterb) / CLOCKS_PER_SEC *1000.0<<" "<<curF<<endl;

	//cout<<"0 "<<actset.size()<<endl;;
	for (i=0;i<db;i++){
		g[i] = shg[i];
	}
	for (i=0; i<db;i++)
		Bdiag[i] = 0.0;
	for (int iter = 0; iter<max_iter; iter++){
		//clock_t titerb = clock();
		M = 0;
		lbfgsm++;
		//cerr << "iter:"<<iter<<" ";
		if (lbfgsm<m){
			matsize = lbfgsm+1;
			dblmiter =2* lbfgsm; //size of dhat, Q and Qhat
		}
		else{
			matsize = m;
			dblmiter =2* m;
		}
		dbmatsize = 2*matsize;
		dhat.clear();
		for (j = 0; j < dblmiter; j++)
			dhat.push_back(0.0);
		for (ii = actset.begin(); ii != actset.end(); ++ii)
			d[*ii] = 0.0;

		//********** find active set **********

		//clock_t tacb = clock();
		oldact = actset;
		//clock_t tace = clock();
		//cerr <<"update oldactset time: "<< (double)(tace - tacb) / CLOCKS_PER_SEC<<endl;

		actset.clear();
		normsg = 0.0;
		for (vector<int>::iterator jj=oldact.begin();jj!=oldact.end();++jj){
			j = *jj;
			if (w[j] <  -1.0e-20){
				subgrad = g[j] - lambda;
				actset.push_back(j);
				normsg += fabs(subgrad); 
				M=fmax(M,fabs(subgrad));
			}
			else if (w[j]> 1.0e-20){
				subgrad = g[j] + lambda;
				actset.push_back(j);
				normsg += fabs(subgrad); 
				M=fmax(M,fabs(subgrad));
			}
			else{
				absg = fabs(g[j])-lambda;
				subgrad = fmax(absg,0.0);
				if (absg > -Mout/n){
					actset.push_back(j);
					normsg += subgrad; 
					M=fmax(M,subgrad);
				}
			}
		}
		//clock_t tnewace = clock();
		//cerr <<"update actset time: "<< (double)(tnewace - tace) / CLOCKS_PER_SEC<<endl;
		if (iter == 0)
			normsg0 = normsg;
		//cerr<<"|act|="<<actset.size()<< " ";
		//termination critiion
		double stopCrit = normsg/normsg0;
		if (stopCrit < epsilon_shrink){
			if (oldact.size() == db && stopCrit < epsilon ){
				//                cerr << "termination criterion attained, iter: "<< iter<<endl;
				/* for (j=0;j<db;j++){
				   cerr<<w[j]<<" ";
				   if (j % 8 ==7)
				   cerr<<endl;
				   }
				   cerr<<endl;
				   */break;
			}
			else {
				actset.clear();
				//clock_t tactb=clock();
				for (j=0;j<db;j++)
					actset.push_back(j);
				//clock_t tacte = clock();
				//cerr <<"reset actset time: "<< (double)(tacte-tactb) / CLOCKS_PER_SEC<<endl;

				//cerr<<"***************************"<<endl;
				Mout = 0.01*lambda*n;

				Ddiag.clear(); L.clear(),Sps.clear();
				for (j=0;j<db;j++){
					//Q[j].clear();
					//Qhat[j].clear();
					Bdiag[j]=0.0;
					Bd[j]=0.0;
				}
				//clock_t tQQe = clock();
				//cerr <<"update Q Qhat time: "<< (double)(tQQe - tacte) / CLOCKS_PER_SEC<<endl;
				prob->grad(actset,shg);
				for(j=0;j<db;j++){
					//shg[j] += u_admm[index] + rho*(w[index]-z_admm[index]);
					index = actset[j];
					shg[j] += prob->augmented->grad(w[index],index);
				}
				for (j=0;j<db;j++){
					g[j] = shg[j] ; //ADMM
				}
				//LR_grad(features,pred,labels,actset,g);
				//clock_t tLRge = clock();
				//cerr <<"update wholeset Grad time: "<< (double)(tLRge - tQQe) / CLOCKS_PER_SEC<<endl;
				lbfgsm = -1;
				epsilon_shrink = epsilon_shrink/10.0;
				if (epsilon_shrink<epsilon)
					epsilon_shrink = epsilon;
				continue;
			}
		}

		/*//debug
		  oldact = actset;
		  actset.clear();
		  for (ii=oldact.begin(); ii != oldact.end() && *ii > 5716672; ++ii)
		  actset.push_back(*ii);
		  *///debug

		Mout = M;
		for (ii = actset.begin(); ii != actset.end(); ++ii){
			Bdiag[*ii] = gamma;
			for(k=0; k< dblmiter; k++)
				Bdiag[*ii] -= Q[*ii][k]*Qhat[*ii][k];
		}
		//for (i=0;i<dhat.size();i++)
		//    dhat[i] = 0;

		//clock_t tinb = clock();
		// actset.clear();                
		//for (j=0;j<db;j++)
		//  actset.push_back(j);

		innerAct = actset; 
		int i_iter = (int)db/actset.size();
		if (i_iter>max_inner)
			i_iter = max_inner;
		//cerr<<"inner_iters="<<i_iter <<" ";
		//*********** inner loop **************
		for (j = 0; j< i_iter; j++){
			shuffle(innerAct);
			for (ii = innerAct.begin(); ii != innerAct.end(); ++ii){ 
				actInd = *ii;
				Bd[actInd] = gamma*d[actInd];
				for (k = 0; k < dblmiter; k++)
					Bd[actInd] -= Q[actInd][k]*dhat[k];


				c = w[actInd]+d[actInd];
				z = -c+softThd(c-(g[actInd]+Bd[actInd])/Bdiag[actInd],lambda/Bdiag[actInd]);
				d[actInd] += z;
				for (k = 0; k < dblmiter; k++)
					dhat[k] += z*Qhat[actInd][k];
			}

		}

		//clock_t tine = clock();

		//double  elapsed =double(tine - tinb) / CLOCKS_PER_SEC; 
		//cerr <<"in: "<<setw(10)<<elapsed<<" ";

		//clock_t tlinb = clock();

		//line search
		alpha = 1.0;

		for (ii=actset.begin(); ii != actset.end(); ++ii){
			j=*ii;
			wtemp[j] = w[j]+alpha*d[j];
		}

		double delta_part = -lambda*l1(w,actset);
		for (ii = actset.begin();ii != actset.end();++ii){
			j = *ii;
			delta_part += d[j]*g[j];
		}

		/*		for (j = 0; j< n; j++)
				delta_pred[j] = 0.0;


				for (ii = actset.begin(); ii != actset.end(); ++ii){
				j = *ii;
				Feature* ins = &(features[j]);
				for (Feature::iterator jj= ins->begin(); jj!=ins->end(); ++jj){
				delta_pred[jj->first] += jj->second*d[j];
				}
				}
				*/
		//cerr<<"d"<<endl;
		prob->compute_fv_change(d,actset,fv_change);

		prob->update_fvalue(fv_change,alpha);
		//for (j = 0; j< n; j++)
		//	cur_pred[j] = pred[j]+alpha*delta_pred[j];
		//clock_t tobjcalb = clock();	

		//candF = prob->fun() + lambda * l1(wtemp,actset) + dot(u_admm,w,actset) + rho/2*( l2(w,actset) + dot(w,z_admm,actset) ); //ADMM
		candF = prob->fun() + lambda * l1(wtemp,actset);
		for(int i=0;i<actset.size();i++) {
			int index = actset[i];
			candF += prob->augmented->fun(wtemp[index],index);
		}
		// l1r_lr_obj(n,db,wtemp,cur_pred,labels,lambda,oldact);
		//clock_t tobjcale = clock();
		//cerr <<"objective calculate time: "<< (double)(tobjcale - tobjcalb) / CLOCKS_PER_SEC<<endl;

		double armijo = alpha*sigma*(lambda*l1(wtemp,actset)+delta_part);
		double alpha_old;
		while ( !(candF <= curF + armijo) ){
			alpha_old = alpha;
			//cerr<<".";
			alpha /= 2.0;
			if (alpha<1e-20){
				cerr<<"alpha is too small "<<endl;
				exit(0);
			}
			prob->update_fvalue(fv_change, alpha - alpha_old);

			for (ii = actset.begin(); ii != actset.end(); ++ii){
				k=*ii;
				wtemp[k] = w[k]+alpha*d[k];
			}
			//candF = prob->fun()+lambda*l1(wtemp,actset)+dot(u_admm,w,actset) + rho/2*( l2(w,actset) + dot(w,z_admm,actset) ); //ADMM
			candF = prob->fun()+lambda*l1(wtemp,actset);
			for(int i=0;i<actset.size();i++) {
				int index = actset[i];
				candF += prob->augmented->fun(wtemp[index],index);
			}
			//l1r_lr_obj(n,db,wtemp,cur_pred,labels,lambda,oldact);
			armijo = alpha*sigma * (lambda * l1(wtemp,actset) + delta_part );
		}
		//cerr<<endl;
		curF = candF;
		//clock_t tline = clock();
		//elapsed =double(tline - tlinb) / CLOCKS_PER_SEC; 
		//cerr <<"line: "<<setw(10)<<elapsed<<" ";

		//clock_t tudb= clock();
		//********** update parameters **************
		// update w,s,t,gamma
		for (ii = actset.begin(); ii != actset.end(); ++ii)
			w[*ii] = wtemp[*ii];
		/*		for (j=0; j<n; j++)
				pred[j] = cur_pred[j];
				*/
		//clock_t tpred = clock();
		//cerr <<"update pred time: "<< (double)(tpred - tudb) / CLOCKS_PER_SEC<<endl;
		prob->grad(actset,shg);
		for(int i=0;i<actset.size();i++){
			int index = actset[i];
			//shg[i] += u_admm[index] + rho*(w[index]-z_admm[index]);
			shg[i] += prob->augmented->grad(w[index],index);
		}
		//LR_grad(features,pred,labels,actset,newg);
		//clock_t tgrad = clock();
		//cerr <<"newg: "<<setw(10)<< (double)(tgrad - tpred) / CLOCKS_PER_SEC<<" ";

		for (int l=0;l<actset.size();l++){ //ii = actset.begin(); ii != actset.end(); ++ii){
			j = actset[l];
			t[j] = shg[l] - g[j];
			s[j] = alpha*d[j];
			g[j] = shg[l];
		}

		double newdiag = 0; 
		gamma = 0;  
		for (ii = actset.begin(); ii != actset.end(); ++ii){
			j = *ii;
			newdiag += s[j] * t[j];
			gamma += s[j] * s[j];
		} 
		gamma = newdiag/gamma; 
		if (newdiag<0)
			cerr<<"s*t< 0 alert,at iter="<<iter<<endl;

		//clock_t tst = clock();
		//cerr <<"add w, gamma, s and t time: "<< (double)(tst - tudb) / CLOCKS_PER_SEC<<endl;

		//update L
		if (lbfgsm < m){
			Sps.push_back(deque<double>());
			L.push_back(deque<double>());
			for (j=0;j<lbfgsm;j++){
				double Lij = 0;
				double Spsij = 0;
				for (ii = actset.begin(); ii != actset.end(); ++ii){
					k = *ii;
					Lij +=s[k]*T[j][k];
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
				qm = (j+lbfgsm) % m;
				double Lij = 0.0;
				double Spsij = 0.0;
				for (ii = actset.begin(); ii != actset.end(); ++ii){
					k = *ii;
					Lij += s[k]*T[qm][k];
					Spsij += s[k]*S[qm][k];
				}   
				L.back().push_back(Lij);
				Sps[j-1].push_back(Spsij);
				Sps.back().push_back(Spsij);
			}
			L.back().push_back(0.0);
			Sps.back().push_back(newdiag/gamma);
		}
		//clock_t tLSps = clock();
		//cerr <<"update L,Sps time: "<< (double)(tLSps - tst) / CLOCKS_PER_SEC<<endl;

		//update S, T and Ddiag
		if (lbfgsm < m){
			for (ii = actset.begin(); ii != actset.end(); ++ii){
				j = *ii;
				S[lbfgsm][j] = s[j];
				T[lbfgsm][j] = t[j];
			}
		}
		if (lbfgsm >= m){
			Ddiag.pop_front();
			qm = lbfgsm % m;
			for (ii = actset.begin(); ii != actset.end(); ++ii){
				j = *ii;
				S[qm][j] = s[j];
				T[qm][j] = t[j];
			}
		}

		//clock_t tadste = clock();
		//cerr <<"add S and T time: "<< (double)(tadste - tLSps) / CLOCKS_PER_SEC<<endl;
		//clock_t test;	
		Ddiag.push_back(newdiag);
		//update Q

		for (ii = actset.begin(); ii != actset.end(); ++ii){
			j = *ii;

			if (lbfgsm<m){
				for (k=0;k<matsize;k++){
					Q[j][k] = gamma*S[k][j];
					Q[j][k+matsize] = T[k][j];
				}
			}
			else{
				for (k=0;k<matsize;k++){
					qm  = (lbfgsm+1+k) % m;
					Q[j][k] = gamma*S[qm][j];
					Q[j][k+matsize] = T[qm][j];
				}
			}
		}
		//clock_t tQe = clock();
		//cerr <<"Q: "<<setw(10)<< (double)(tQe - tadste) / CLOCKS_PER_SEC<<" ";

		//copmute R and update Qhat
		MatrixXd invr=MatrixXd::Zero(dbmatsize,dbmatsize);
		for (int i=0;i<matsize;i++ ){
			for (int j=0;j<matsize;j++){
				invr(i,j) = gamma * Sps[i][j];
			}
		}
		MatrixXd matL(matsize,matsize);
		for (int i=0;i<matsize;i++ )
			for (int j=0;j<matsize;j++)
				matL(i,j) = L[i][j];
		for (int i=0;i<matsize;i++)
			invr(i+matsize,i+matsize) = - Ddiag[i];
		invr.topRightCorner(matsize,matsize) = matL;
		invr.bottomLeftCorner(matsize,matsize) = matL.transpose();
		MatrixXd R(dbmatsize,dbmatsize);

		//clock_t tinvb = clock();
		//cerr <<"update invR time: "<< (double)(tinvb-tQe) / CLOCKS_PER_SEC<<endl;

		R = invr.inverse();
		//VectorXcd eivals = R.eigenvalues();
		//cout << "The eigenvalues of R are:" << endl << eivals << endl;

		//clock_t tinve = clock();
		//cerr <<"inverse time: "<< (double)(tinve - tinvb) / CLOCKS_PER_SEC<<endl;

		double temp;
		for (ii = actset.begin(); ii != actset.end(); ++ii){
			i = *ii;
			for (int j=0;j<dbmatsize;j++){
				temp=0.0;
				for (int k=0;k<dbmatsize;k++)
					temp += Q[i][k]*R(k,j);
				Qhat[i][j] = temp;
			}
		}

		//clock_t tude = clock();
		//cerr <<"Qhat: "<< setw(10)<<(double)(tude - tinve) / CLOCKS_PER_SEC<<" ";

		//cerr <<"total update time: "<< (double)(tude - tudb) / CLOCKS_PER_SEC<<endl;
		//cerr <<"one iter time: "<< (double)(tude - titerb) / CLOCKS_PER_SEC<<endl;
		//       cerr<<(double)(tude - touterb) / CLOCKS_PER_SEC *1000.0<<" ";
		//cerr <<(double)(tude - touterb) / CLOCKS_PER_SEC *1000.0<<" "<<curF<<endl;
		//cout<<(double)(tude - touterb) / CLOCKS_PER_SEC *1000.0<<" "<<actset.size()<<endl;;

		}
		//clock_t touterend = clock();
		/* for (int i=0;i<db;i++){
		   cerr<<" "<<w[i];
		   if (i%8 == 7)
		   cerr<<endl;
		   }    
		   */ 	//cerr <<"outer iter total time: "<< (double)(touterend - touterb) / CLOCKS_PER_SEC<<endl;

		//	delete [] pred;
		delete [] g;
		delete [] newg;
		for (i=0;i<m;i++)
			delete [] S[i];
		delete [] S; 
		for (i=0;i<m;i++)
			delete [] T[i];
		delete [] T;
		//	delete [] t;
		//	delete [] s;
		delete [] Bdiag;
		delete [] Bd;
		delete [] d;
		delete [] wtemp;
		//	delete [] delta_pred;
		//  delete [] cur_pred;
		//delete deque<double> Ddiag,dhat;
		//delete qmat Q,Qhat,S,T,L;
	}

