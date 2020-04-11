import numpy as np
import argparse
from matplotlib import pyplot as plt
rewards = []
EPOSIDES = 250
lineStyle = ['-b','--r','.g']
def plot(f, arr, strLabel):

    strLine = f.readline()
    start = strLine.find('INFO')
    if start != -1:
        start += len('INFO:')
        tittle = strLine[start:-1]
    else :
        tittle = 'tittle'
    while True:
        strLine = f.readline()
        if strLine == '':
            break

        start = strLine.find('reward')
        if start != -1:
            start += len('reward:')
            rewards.append(strLine[start:-1])
    f.close()
    y1 = np.asarray(rewards, dtype=np.float)

    rewardLen = len(rewards)
    x = np.arange(0,rewardLen)

    plt.title(tittle)
    plt.xlabel("eposides")
    plt.ylabel("rewards")
    i = arr % 3
    plt.plot(x,y1,lineStyle[i],label= strLabel)
    plt.legend()

    rewards.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot from log")
    parser.add_argument('--file','-f', action='append', dest='files', help='input the log file',type=argparse.FileType('r'))
    args = parser.parse_args()
    num =len(args.files)
    i=0
    for f in args.files:
        plot(f, i, f.name)
        f.close
        i+=1
    plt.show()




