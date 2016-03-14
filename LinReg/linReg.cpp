#include "linReg.h"

void read_data(char* fname, Matrix& X, Matrix& y, int& N, int& D){

	ifstream fin(fname);
	fin >> N >> D;
	
	X = Matrix(N,D);
	y = Matrix(N,1);
	
	float label, fea;
	for(int i=0;i<N;i++){
		
		fin >> label;
		y << label;
		
		for(int j=0;j<D;j++){
			fin >> fea;
			X << fea;  //row-majorly fill in
		}
	}
}

void solve_w( Matrix& X, Matrix& y,   Matrix& w ){

	Matrix X_t = MatrixMath::Transpose(X);
	Matrix cov_mat = X_t * (X); 
	w = MatrixMath::Inv( cov_mat ) * X_t * y;
}

void solve_w_aug( Matrix& X, Matrix& y, Matrix& u, Matrix& z, float rho,  Matrix& w ){

	// X'X
	int N = X.getRows();
	int D = X.getCols();
	Matrix X_t = MatrixMath::Transpose(X);
	
	// cov_mat = (X'X + \rho * I)
	Matrix cov_mat = X_t * (X) + rho*MatrixMath::Eye(D); 
	
	w = MatrixMath::Inv( cov_mat ) * ( X_t*y - u + rho*z );
}


void LinRegSubsolver::readData(char* fname){

	read_data(fname, X,y, N, D);
}

int LinRegSubsolver::model_dim(){
	return D;
}


void LinRegSubsolver::subSolve(double* u, double* z, double rho, double* model){

	//In this case, w == model
	Matrix u_mat(D,1);
	Matrix z_mat(D,1);
	for(int i=0;i<D;i++){
		u_mat << u[i];
		z_mat << z[i];
	}
	
	//solve
	Matrix w(D,1);
	solve_w_aug(X,y,u_mat,z_mat, (float)rho,  w);
	
	//parse
	for(int i=0;i<D;i++){
		model[i] = (double)w(i+1,1);
	}
}


void LinRegSubsolver::writeModel(char* fname, double* model){	
	
	ofstream model_out(fname);
	model_out << D << endl;
	for(int i=0;i<D;i++){
		model_out << model[i] << endl;
	}
	model_out.close();
}

/*
int main(int argc, char** argv){

	if(argc < 2){
		cerr << "./linReg  [train_data]" << endl;
		exit(0);
	}
	
	char* data_file = argv[1];
	
	Matrix X;
	Matrix y;
	int N,D;
	read_data(data_file, X,y, N, D);
	cerr << N << ", " << D << endl;
	
	Matrix w;
	//Matrix u(D,1), z(D,1);
	//for(int i=0;i<D;i++){
	//	u << 0.0;
	//	z << 0.0;
	//}
	
	//double rho=10.0;
	//solve_w_aug( X, y, u, z, rho, w );
	solve_w( X, y,  w);
	w.print();
	
	return 0;
}
*/
