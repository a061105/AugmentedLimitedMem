#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include "proxGrad.h"

using namespace std;
proxGrad::proxGrad(){};

void proxGrad::minimize(Problem* prob){
	int d = prob->d;
cerr<<d<<endl;	
	double* w = prob->w;
	double beta = 0.5;
    
	
	vector<int> act_set;
	for(int i=0;i<d;i++)
		act_set.push_back(i);
	
	vector<double> grad;
	

//	cerr << "initial obj=" << h+f << endl;
	
	double* w_new = new double[d];
	for(int i=0;i<d;i++)
		w_new[i] = w[i];

	double* w_change = new double[d];
	vector<pair<int,double> > fv_change;
	int iter = 0;
	double f_new, h_new;
    clock_t touterb = clock();;
	double h = lambda*l1_norm(w, d);
	double f = prob->fun();
		
    clock_t firsti = clock();
    cerr<<std::setprecision(20)<<(double)(firsti - touterb) / CLOCKS_PER_SEC *1000.0<<" ";
    cerr <<" "<<f+h<<endl;

    cout<<std::setprecision(20)<<(double)(firsti - touterb) / CLOCKS_PER_SEC *1000.0<< " "<<d<<endl;
    
    while( iter < max_iter ){
		
		//compute gradient
		//cerr <<"grad" << endl;
		prob->grad( act_set,    grad );
		
		//line search
		int r;
	    double t = 1.0;
		while(1){
			
			//grad descent step
			for(int j=0;j<act_set.size();j++){
				r = act_set[j];
				w_new[r] = w[r] - t*grad[j];
			}
			//proxiaml step
			softThd( w_new, d, t*lambda );
			
			h_new = h;
			for(int j=0;j<act_set.size();j++){
				r = act_set[j];
				w_change[r] = w_new[r] - w[r];
				h_new += lambda*(fabs(w_new[r]) - fabs(w[r]));
			}
			
			//check sufficient descent condition
			
			//real decrease
			//cerr << "compute_fv" << endl;
			prob->compute_fv_change(w_change, act_set,    fv_change);
			//cerr << "infer" << endl;
			prob->update_fvalue(fv_change, 1.0);
			//cerr << "fun" << endl;
			f_new = prob->fun();
			
			if( h_new+f_new < h+f ){
				
				for(int j=0;j<act_set.size();j++){
					r = act_set[j];
					w[r] = w_new[r];
				}
				break;
			}else{
				prob->update_fvalue(fv_change, -1.0);
				t = beta*t;
				
				//cerr << h_new + f_new << " back to " << h + prob->fun() << endl;

				if( t < 1e-15 ){
					cerr << "t=" << t << endl;
					/*for(int i=0;i<prob->numLabel;i++){
						cerr << prob->label_name_map.find(i)->second << "\t";
						for(int j=0;j<prob->numLabel;j++){
							cerr << grad[prob->bi_offset_w+i*prob->numLabel+j]  << "\t";
						}
						cerr << endl;
					}
					cerr << "p(y|y)" << endl;
					for(int i=0;i<prob->numLabel;i++){
						cerr << prob->label_name_map.find(i)->second << "\t";
						for(int j=0;j<prob->numLabel;j++){
							cerr << prob->factor_yy[i*prob->numLabel+j]  << "\t";
						}
						cerr << endl;
					}*/
					delete[] w_new;
					delete[] w_change;
					
					return ;
				}
			}
		}
		f = f_new;
		h = h_new;
		
		if(iter%1==0){
            clock_t tude = clock();
            double tm =(double)(tude - touterb) / CLOCKS_PER_SEC *1000.0; 
	    cerr<<tm;
            cerr <<" "<<f+h<<endl;
	    int nnz=0;
	    for (int i=0;i<d;i++)
		    if(fabs(w[i])>1e-20)
			    nnz++;
	    cout<<tm<<" "<<nnz<<endl;;
        }
            //cerr << "iter=" << iter << ", obj=" << f+h << ", t=" << t <<  endl;
		iter++;
	}
	
	/*for(int i=0;i<prob->numLabel;i++){
		cerr << prob->label_name_map.find(i)->second << "\t";
		for(int j=0;j<prob->numLabel;j++){
			cerr << grad[prob->bi_offset_w+i*prob->numLabel+j]  << "\t";
		}
		cerr << endl;
	}
	cerr << "p(y|y)" << endl;
	for(int i=0;i<prob->numLabel;i++){
		cerr << prob->label_name_map.find(i)->second << "\t";
		for(int j=0;j<prob->numLabel;j++){
			cerr << prob->factor_yy[i*prob->numLabel+j]  << "\t";
		}
		cerr << endl;
	}*/
	
	delete[] w_new;
	delete[] w_change;
}


