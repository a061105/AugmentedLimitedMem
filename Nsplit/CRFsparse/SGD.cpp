#include "SGD.h"
#include <iostream>
#include<cmath>
using namespace std;


void SGD::minimize(Problem_SGD* prob){
	
	int d = prob->d;
	double* w = prob->w;
	int N = prob->N;
	vector<pair<int,double> > grad;
    vector<pair<int,double> >::iterator it;
	double h = lambda*l1_norm(w, d);
	double f = prob->fun();
	
	//cerr << "initial obj=" << h+f << endl;
	
	vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	shuffle(index);
	
	int iter = 0;
    double eta;
    u = 0.0;
    double z;
    q = new double[d];
    for (int i=0;i<d;i++)
        q[i]=0.0;
    clock_t tstart =clock();
    //int j;
    f = prob->fun();
    h = lambda * l1_norm( w , d );

    double t =1000.0*(double)(clock()-tstart)/CLOCKS_PER_SEC;  
    cerr<<std::setprecision(15) << t <<" "<<f+h<<endl;	
    while( iter < max_iter ){
	    for(int k=0;k<N;k++){
		    int i = index[k];
		    //   eta = eta0;
		    eta = eta0/(1.0+((double)N*iter+k)/(double)N);
		    //eta = eta0*pow(alpha,(double)(iter+double(k)/N));
            u += eta*lambda/N;
			//compute gradient
			prob->applyGrad( i,   eta, this );
			//preInd = -1;
            //update
//				cerr << w[j] << ", " << grad[j] << endl;
/*			for(it=grad.begin();it!=grad.end();it++){
                w[it->first] =  w[it->first] - eta * it->second;
                if (preInd != it->first){
                    //penalty();
                    j = it->first;
                    z = w[j];
                    if (w[j]>0)
                        w[j] = fmax(0.0,z-u-q[j]);
                    else if(w[j]<0)
                        w[j] = fmin(0.0,z+u-q[j]);
                    q[j] += w[j]-z;
                }
                preInd = it->first;
            } */

		}
        //cerr<<" eta="<<eta<<" ";

		if(iter%5==4){
			f = prob->fun();
			h = lambda * l1_norm( w , d );
			
	    		double t =1000.0*(double)(clock()-tstart)/CLOCKS_PER_SEC;  
		    	if (t>3*3600000) exit(0);
			cerr<<std::setprecision(15) << t <<" "<<f+h<<endl;	
            //cerr << "iter=" << iter << ", obj=" << f+h << ", h=" << h << ", f=" << f << " time elapsed="<<(double)(clock()-tstart)/CLOCKS_PER_SEC <<endl;
		    int nnw = 0;
            for (int j = 0;j<d;j++){
                if (fabs(w[j])>1e-12)
                    nnw++;
            }
            cout <<std::setprecision(15)<< iter <<" "<<nnw<<endl;
        }

		iter++;
		shuffle(index);
	}
    //delete [] q;
}

void SGD::penalty(int j,double *w){
    double z = w[j];
    if (w[j]>0)
        w[j] = fmax(0.0,z-u-q[j]);
    else if(w[j]<0)
        w[j] = fmin(0.0,z+u-q[j]);
    q[j] += w[j]-z;
}
