#include "problem.h"

//vectoractset_owl
double Problem::Eval(vector<double> &inputw, vector<double> & g){
	double *w_change = new double[d];
	vector<pair<int,double> > fv_change;
	for (int i=0;i<d;i++){
		w_change[i] = inputw[i] - w[i];
	}
	//cerr<<"w "<<w[0]<<endl;
	compute_fv_change(w_change,wholeset,fv_change);
	update_fvalue(fv_change,1.0);

	grad(wholeset,g);
	for (int i=0;i<d;i++)
		w[i] = inputw[i];

	delete [] w_change;
	return fun();
}
