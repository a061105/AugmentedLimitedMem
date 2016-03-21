CXX ?= g++
CFLAGS = -O3 
#-std=c++0x -static-libstdc++
LIBS = ../lib/
PROB_OBJ = taxonomy.o  seq_label.o multiclass.o pseudo_seqlabel.o proxQN.o  util.o 

all: clean predict train

train: $(PROB_OBJ) train.cpp
	$(CXX) $(CFLAGS)  $(PROB_OBJ) train.cpp  -o train

predict:  $(PROB_OBJ) predict.cpp
	$(CXX) $(CFLAGS)  $(PROB_OBJ) predict.cpp  -o predict

proxQN.o: 
	$(CXX) $(CFLAGS) -I$(LIBS) -c proxQN.cpp -o proxQN.o

seq_label.o: util.o
	$(CXX) $(CFLAGS) -c seq_label.cpp -o seq_label.o

multiclass.o: util.o
	$(CXX) $(CFLAGS) -c multiclass.cpp -o multiclass.o

taxonomy.o: util.o
	$(CXX) $(CFLAGS) -c taxonomy.cpp -o taxonomy.o

pseudo_seqlabel.o: util.o
	$(CXX) $(CFLAGS) -c pseudo_seqlabel.cpp -o pseudo_seqlabel.o

util.o:
	$(CXX) $(CFLAGS) -c util.cpp -o util.o

clean:
	rm -f *.o *~ predict train
