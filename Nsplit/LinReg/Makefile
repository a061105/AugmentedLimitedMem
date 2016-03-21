flag = -O3 -w

all: linReg.o Matrix.o MatrixMath.o Operators.o

linReg.o: linReg.cpp
	g++ $(flag) -c -o linReg.o linReg.cpp

Matrix.o: Matrix.cpp
	g++ $(flag) -c -o Matrix.o Matrix.cpp

MatrixMath.o: MatrixMath.cpp
	g++ $(flag) -c -o MatrixMath.o MatrixMath.cpp

Operators.o: Operators.cpp
	g++ $(flag) -c -o Operators.o Operators.cpp

#g++ -O3 -o linReg linReg.cpp Matrix.cpp MatrixMath.cpp Operators.cpp
