#include "BCD.h"
#include <iostream>
#include "util.h"

using namespace std;
double l1_norm(double* w, vector<int> &act_set){
    double l1=0.0;    
    for (vector<int>::iterator it=act_set.begin(); it != act_set.end(); it++)
        l1 += fabs(w[*it]);
    return l1;
}

void BCD::minimize(Problem* prob){
    double *w = prob->w;
    int d= prob->d;
    int n = prob->N;
    double *new_w = new double[d];
    double *delta_w = new double[d];
    double *d_w = new double[d];
    for (int i=0;i<d;i++){
        new_w[i] = w[i];
        delta_w[i]= 0.0;
        d_w[i] = 0.0;
    }
    vector<int>* blks = prob->buildBlocks();
    int numBlks = prob->numBlocks;
    vector<int>* act_sets = new vector<int>[numBlks];
    for (int b=0;b<numBlks;b++)
        act_sets[b] = blks[b];
    double epsilon_shrink = 1000.0*epsilon;
    if (epsilon_shrink<0.1)
        epsilon_shrink = 0.1;

    vector<double> gi;
    vector<double> hii;
    //double *gi = new double[d/numBlks];
    //double *hii =new double[d/numBlks];
    
    int coordinate;
    double psudograd;
    vector<int> blockOrder;
    for (int b=0;b<numBlks;b++)
        blockOrder.push_back(b);
    //vector<pair<int,double> > fv_change;
    //vector<double> xg,xh;
    double candF;
    double sigma = 0.001;
    double alpha,alpha_old,armijo, hblock_new,hblock_old;;
	clock_t tstart = clock();
    double f=prob->fun();
    double h=0;
    double curF = f;
    cerr<<"fun obj0="<<curF<<endl;
    double t =1000.0*(double)(clock()-tstart)/CLOCKS_PER_SEC;  
    cerr<<std::setprecision(15) << t <<" "<<curF<<endl;	
    vector<int> oldAct;
    double delta_h = 0.0;
	double normsg0,normsg,M,Mout,absg,subgrad;
    int nnz=0;
    Mout = 0.01*lambda*n;
    int startover = 2;
    for (int iter = 0;iter<max_iter;iter++){
        M = 0.0;
        startover -= 1;
		normsg = 0.0;
        nnz = 0;
//        cerr<<"iter="<<iter<<endl;
        for (int bb=0;bb<numBlks;bb++){
            int b = blockOrder[bb];
            oldAct = act_sets[b];
  //          cerr<<"beginGrad"<<endl;
            
            prob->derivatives(oldAct,gi,hii);
    //        cerr<<"endGrad"<<endl;
            armijo = 0.0;
            act_sets[b].clear();
            for (int i=0;i<oldAct.size();i++){
                coordinate = oldAct[i];
                if (w[coordinate] <  -1.0e-20){
                    subgrad = gi[i] - lambda;
                    act_sets[b].push_back(coordinate);
                    normsg += fabs(subgrad); 
                    M=fmax(M,fabs(subgrad));
                }
                else if (w[coordinate]> 1.0e-20){
                    subgrad = gi[i] + lambda;
                    act_sets[b].push_back(coordinate);
                    normsg += fabs(subgrad); 
                    M=fmax(M,fabs(subgrad));
                }
                else{
                    absg = fabs(gi[i])-lambda;
                    subgrad = fmax(absg,0.0);
                    if (absg > -Mout/n){
                        act_sets[b].push_back(coordinate);
                        normsg += subgrad; 
                        M=fmax(M,subgrad);
                    }
                    else 
                        continue;
                }               
      //          cerr<<"nnz="<<nnz<<endl;
                if (hii[i]<1.0e-10){
                    hii[i] = 1.0e-10;
                }
                
                psudograd = hii[i] * w[coordinate]-gi[i]; 
                new_w[coordinate] = softThd(psudograd,lambda)/(hii[i]);
        //        cerr<<"hello1"<<endl;
                delta_w[coordinate] = new_w[coordinate] - w[coordinate];
                d_w[coordinate] = delta_w[coordinate];
                armijo += gi[i]*d_w[coordinate]; 
          //      cerr<<"hello1.5"<<endl;
                //w[coordinate] = new_w[coordinate];

            }
            //cerr<<"updateMesg"<<endl;
            prob->update_Mesg(act_sets[b],delta_w);
            //cerr<<"finish updateMesg"<<endl;
            nnz += act_sets[b].size();
            
            //line search
            alpha = 1.0; 
            f = prob->fun();
            hblock_old =l1_norm(w,act_sets[b]); 
            hblock_new =l1_norm(new_w,act_sets[b]); 
            delta_h =lambda*(hblock_new - hblock_old); 
            h += delta_h;
            hblock_old = hblock_new;
            armijo = sigma * (armijo+delta_h);
            candF= f+h;
            while ( !(candF <= curF + alpha*armijo) ){
                alpha_old = alpha;
                //cerr<<".";
                alpha /= 2.0;
                if (alpha<1e-20){
//                    cerr<<"alpha is too small "<<endl;
                    break;
                    //exit(0);
                }
                for (vector<int>::iterator it=act_sets[b].begin();it != act_sets[b].end();it++){
                    coordinate = *it;
                    delta_w[coordinate] = (alpha-alpha_old)*d_w[coordinate];
                    new_w[coordinate] += delta_w[coordinate];
                }
                prob->update_Mesg(act_sets[b],delta_w);
                f = prob->fun();
                hblock_new =l1_norm(new_w,act_sets[b]);  
                h += lambda*(hblock_new - hblock_old);
                hblock_old = hblock_new;
                candF = f+h;
            }
            curF = candF;
            for (vector<int>::iterator it=act_sets[b].begin();it != act_sets[b].end();it++){
                delta_w[*it] = 0.0;
                w[*it] = new_w[*it];
            }
        }
        //cerr<<"out of outer iteration"<<endl;
        if (iter == 0)
            normsg0 = normsg;
        Mout = M;
        double stopCrit = normsg/normsg0;
        //cerr<<"stopCrit="<<stopCrit<<endl;
        if (stopCrit < epsilon_shrink){
            if (startover == 1 && stopCrit < epsilon ){
                cerr << "termination criterion attained, iter: "<< iter<<endl;
                break;
            }
            else {
                for (int b=0;b<numBlks;b++)
                    act_sets[b] = blks[b];

//                cerr<<"***************************"<<endl;
                Mout = 0.01*lambda*n;
                startover = 2;
                epsilon_shrink = epsilon_shrink/10.0;
                if (epsilon_shrink<epsilon)
                    epsilon_shrink = epsilon;
            }
        }
        t =1000.0*(double)(clock()-tstart)/CLOCKS_PER_SEC;  
        cerr<<std::setprecision(15) << t <<" "<<curF<<endl;	
        cout<<std::setprecision(15) << t <<" "<<nnz<<endl;	
        //shuffle(blockOrder);
        //cerr<<setprecision(15)<<"iter="<<iter<<" obj="<<prob->fun()+lambda*l1_norm(w,d)<<endl;
    }
}
