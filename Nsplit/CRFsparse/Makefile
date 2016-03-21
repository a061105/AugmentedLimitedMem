CXX ?= g++
CFLAGS = -O3 -w -std=c++0x -static-libstdc++
LIBS = /usr/include/eigen3/
PROB_OBJ = util.o problem.o  BCD.o taxonomy.o   seq_label.o proxGrad.o proxLBFGS.o  OWLQN.o 
#seq_label_sgd.o SGD.o taxonomy_sgd.o

all:  clean train

train: $(PROB_OBJ)
	$(CXX) $(CFLAGS) -c $(PROB_OBJ)  CRFsparse.cpp  -o  CRFsparse.o

OWLQN.o: util.o
	$(CXX) $(CFLAGS) -c OWLQN.cpp -o OWLQN.o

#SGD.o:
#	$(CXX) $(CFLAGS) -c SGD.cpp -o SGD.o

BCD.o:
	$(CXX) $(CFLAGS) -c BCD.cpp -o BCD.o
	
proxLBFGS.o: 
	$(CXX) $(CFLAGS) -I$(LIBS) -c proxLBFGS.cpp -o proxLBFGS.o

proxGrad.o:
	$(CXX) $(CFLAGS) -c proxGrad.cpp -o proxGrad.o

#seq_label_sgd.o: util.o
#	$(CXX) $(CFLAGS) -c seq_label_sgd.cpp -o seq_label_sgd.o

seq_label.o: util.o
	$(CXX) $(CFLAGS) -c seq_label.cpp -o seq_label.o

taxonomy.o: util.o
	$(CXX) $(CFLAGS) -c taxonomy.cpp -o taxonomy.o

#taxonomy_sgd.o: util.o
#	$(CXX) $(CFLAGS) -c taxonomy_sgd.cpp -o taxonomy_sgd.o

util.o:
	$(CXX) $(CFLAGS) -c util.cpp -o util.o

problem.o:
	$(CXX) $(CFLAGS) -c problem.cpp -o problem.o
clean:
	rm -f *.o
