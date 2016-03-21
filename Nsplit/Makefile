flag = -O3 -w -std=c++0x -static-libstdc++

LIBS = Liblinear

LIBS_obj = Liblinear/*.o Liblinear/blas/blas.a CRFsparse/*.o #SMIDAS/*.o

all: ADMM 

ADMM: $(LIBS) ADMMseq.cpp
	g++ $(flag) -o ADMMseq ADMMseq.cpp Subsolver.cpp util.cpp ADMMAug.cpp DDAug.cpp proxAug.cpp $(LIBS_obj)

LinReg: 
	cd LinReg; make; cd ../
Liblinear:
	cd Liblinear; make -B; cd ../
CRFsparse:
	cd  CRFsparse; make -B; cd ../
#SMIDAS:
#	cd  SMIDAS; make -B; cd ../
split:
	g++ $(flag) -o split split.cpp util.cpp Subsolver.cpp ADMMAug.cpp $(LIBS_obj)

clean:
	rm -rf tmp_info.*
