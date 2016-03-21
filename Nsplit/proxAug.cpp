#include "proxAug.h"
#include "Liblinear/train.h"
#include <cstdio>
#include <vector>

proxAugmented::proxAugmented() {
	w_init = NULL;
}
double proxAugmented::fun(double* w) {return 0;}
void proxAugmented::grad(double* w, double* g) {return;}
void proxAugmented::Hs(double* w, double* Hs) {return;}
double* proxAugmented::getModel() {
	return w_init;
}
double proxAugmented::fun(double w, int index) {
	return w*mu[index]+0.5*eta_t*(w-wt[index])*(w-wt[index]);
}
double proxAugmented::grad(double w, int index) {
	return mu[index]+eta_t*(w-wt[index]);
}
double proxAugmented::Hii(double w, int index){
	return eta_t;
}
double* proxAugmented::getMu_B() {
	return mu;
}
void proxAugmented::getAlpha(double* alpha) {
	for(int i=0;i<instance.size();i++) {
		alpha[i] = global_alpha[instance[i]];
	}
	return;
}
void proxAugmented::setAlpha(double* alpha) {
	for(int i=0;i<instance.size();i++)
		global_alpha[instance[i]] = alpha[i];
	return;
}
