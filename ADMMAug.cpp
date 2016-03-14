#include "ADMMAug.h"
#include "Liblinear/train.h"
#include <cstdio>
#include <vector>

ADMMAugmented::ADMMAugmented() {
	mu = NULL;
	z = NULL;
	rho = 0;
	w_init = NULL;
}
ADMMAugmented::ADMMAugmented(double* _mu, double* _z, double _rho, double* _w_init) {
	mu = _mu;
	z = _z;
	rho = _rho;
	w_init = _w_init;
}
double ADMMAugmented::fun(double* w) {return 0;}
void ADMMAugmented::grad(double* w, double* g) {return;}
void ADMMAugmented::Hs(double* w, double* Hs) {return;}
double* ADMMAugmented::getModel() {
	return w_init;
}
double ADMMAugmented::fun(double w, int index) {
	return mu[index]*(w-z[index])+0.5*rho*(w-z[index])*(w-z[index]);
}
double ADMMAugmented::grad(double w, int index) {
	return mu[index]+rho*(w-z[index]);
}
double ADMMAugmented::Hii(double w, int index){
	return rho;
}
double* ADMMAugmented::getMu_B() {
	return NULL;
}
void ADMMAugmented::getAlpha(double* alpha) {
	alpha = NULL;
	return;
}
void ADMMAugmented::setAlpha(double* alpha) {
	return;
}
