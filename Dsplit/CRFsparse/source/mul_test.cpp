
typedef vector<pair<int,double> > Feature;

int main(){
	
	Feature x;
	x.push_back(make_pair(3,3));
	x.push_back(make_pair(3,3));
	x.push_back(make_pair(3,3));
	x.push_back(make_pair(3,3));

	double** W = new double*[K];//  K by D
	for(int k=0;k<K;k++)
		W[k] = new double[D];
		
	//multiply
	double* fxy = new double[K];

	for(int k=0;k<K;k++){
		fxy[k] = 0.0;
		for(Feature::iterator it=x.begin(); it!=x.end();it++)
			fxy[k] += W[k][ it->first ]  * it->second;
	}
}
