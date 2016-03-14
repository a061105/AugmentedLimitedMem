#include "DDAug.h"
#include "Liblinear/train.h"
#include <cstdio>
#include <vector>

DDAugmented::DDAugmented() {
	w_init = NULL;
}
DDAugmented::DDAugmented(double* _w_init) {
	w_init = _w_init;	
}
double DDAugmented::fun(double* w) {return 0;}
void DDAugmented::grad(double* w, double* g) {return;}
void DDAugmented::Hs(double* w, double* Hs) {return;}
double* DDAugmented::getModel() {
	return w_init;
}
double DDAugmented::fun(double w, int index) {
	return w*mu[index];
}
double DDAugmented::grad(double w, int index) {
	return mu[index];
}
double DDAugmented::Hii(double w, int index){
	return 0;
}
double* DDAugmented::getMu_B() {
	return mu;
}
void DDAugmented::getAlpha(double* alpha) {
	for(int i=0;i<instance.size();i++) {
		alpha[i] = global_alpha[instance[i]];
	}
	return;
}
void DDAugmented::setAlpha(double* alpha) {
	for(int i=0;i<instance.size();i++)
		global_alpha[instance[i]] = alpha[i];
	return;
}
