import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def main():
    fp = open('aloi.bin.lib','r')
    time = []
    fun = []
    for line in fp:
        line = line.strip().split(' ')
        time.append(line[0])
        fun.append(line[1])

    fp = open('aloi.bin.BCDwithaug.15','r')
    time1 = []
    fun1 = []
    for line in fp:
        line = line.strip().split(' ')
        time1.append(line[0])
        fun1.append(line[1])

    fp = open('aloi.bin.BCD','r')
    time2 = []
    fun2 = []
    for line in fp:
        line = line.strip().split(' ')
        time2.append(line[0])
        fun2.append(line[1])

    plt.semilogx(time,fun,label='liblinear')
    plt.semilogx(time1,fun1,label='Aug-BCD')
    plt.semilogx(time2,fun2,label='BCD')
    plt.legend(loc=2)
    plt.ylabel('objective function')
    plt.xlabel('time(s)')
    plt.savefig('aloi.png')
if __name__ == '__main__':
    main()
