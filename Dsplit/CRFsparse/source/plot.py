import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def main():
    fp = open('data.proxQN.LSHTC1','r')
    time = []
    fun = []
    for line in fp:
        line = line.strip().split('\t')
        time.append(line[0])
        fun.append(line[1])
    fp = open('data.BCD.lshtc1','r')
    time1 = []
    fun1 = []
    for line in fp:
        line = line.strip().split('\t')
        time1.append(line[0])
        fun1.append(line[1])
    fp = open('data.BCD.5.lshtc1','r')
    time2 = []
    fun2 = []
    for line in fp:
        line = line.strip().split('\t')
        time2.append(line[0])
        fun2.append(line[1])
    plt.plot(time,fun,label='proxQN')
    plt.plot(time1,fun1,label='BCD m=1')
    plt.plot(time2,fun2,label='BCD m=5')
    plt.legend(loc=1)
    plt.ylabel('objective function')
    plt.xlabel('time(s)')
    plt.savefig('lshtc1.png')
if __name__ == '__main__':
    main()
