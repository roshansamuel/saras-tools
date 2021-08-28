import h5py as hp
import numpy as np
from matplotlib import cm
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

# Pyplot-specific directives
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["font.weight"] = "medium"

# Available variables for plotting
nArray = ['xVel', 'yVel', 'zVel', 'Pres', 'Temp']

# Choose variable to be plotted as the index corresponding to it in above arrays
plotVar = 4

def loadData(fileName):
    global plotVar
    global U, V, W, P, T, X, Y, Z

    print("Reading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    U, V, W, P, T = 1, 1, 1, 1, 1
    if plotVar == 0:
        U = np.array(sFile["Vx"])
    elif plotVar == 1:
        V = np.array(sFile["Vy"])
    elif plotVar == 2:
        W = np.array(sFile["Vz"])
    elif plotVar == 3:
        P = np.array(sFile["P"])
    elif plotVar == 4:
        T = np.array(sFile["T"])

    X = np.array(sFile["X"])
    Y = np.array(sFile["Y"])
    Z = np.array(sFile["Z"])

    sFile.close()


def plotFile(iStr, iEnd):
    global vNum
    global U, V, W, P, T, X, Y, Z

    vArray = [U, V, W, P, T]
    X1, Z1 = np.meshgrid(X, Z)
    for i in range(iStr, iEnd):
        fig = plt.figure(figsize=(20, 8))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_aspect("auto")

        r = [0, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="b")

        planeY = i/1024
        a = [x for x in combinations(np.array(list(product(r, r))), 2) if np.sum(np.abs(x[0]-x[1])) == r[1]-r[0]]
        b = [(np.array([x[0][0], planeY, x[0][1]]), np.array([x[1][0], planeY, x[1][1]])) for x in a]
        for s, e in b:
            ax.plot3D(*zip(s, e), color='r')

        ax = fig.add_subplot(1, 2, 2)
        im = ax.contourf(X1, Z1, vArray[vNum][:, i, :].transpose(), cmap=cm.jet, levels=1000, vmin=0.3, vmax=0.7)

        fig.subplots_adjust(right=0.8)
        cb = fig.colorbar(im)
        cb.ax.tick_params(labelsize=20)

        ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
        ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)

        plt.tight_layout()
        plt.savefig(nArray[vNum] + "_Vert_{0:04d}.png".format(i))
        plt.close()

    return 0


def main(sTime):
    global vNum

    vNum = 4

    Nx = 1024
    nProcs = 12

    fileName = "Soln_{0:09.4f}.h5".format(sTime)
    loadData(fileName)

    pool = mp.Pool(processes=nProcs)

    rangeDivs = [int(x) for x in np.linspace(0, Nx, nProcs+1)]
    rangeList = [(rangeDivs[x], rangeDivs[x+1]) for x in range(nProcs)]

    poolRes = [pool.apply_async(plotFile, args=(x[0], x[1])) for x in rangeList]
    cosVals = [x.get() for x in poolRes]


main(933.5)
