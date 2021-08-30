import h5py as hp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# Pyplot-specific directives
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["font.weight"] = "medium"

# Set these times according to the solution generation script
inpTime = 675
outTime = 320

# Variable array
nArray = ['xVel', 'yVel', 'zVel', 'Pres', 'Temp']

def loadData(fileName):
    print("Reading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    U = np.array(sFile["Vx"])
    V = np.array(sFile["Vy"])
    W = np.array(sFile["Vz"])
    P = np.array(sFile["P"])
    T = np.array(sFile["T"])

    X = np.array(sFile["X"])
    Y = np.array(sFile["Y"])
    Z = np.array(sFile["Z"])

    sFile.close()

    return X, Y, Z, U, V, W, P, T


def plotHorz(vNum):
    global nArray
    global inpTime, outTime

    fileName = "Soln_{0:09.4f}.h5".format(inpTime)
    x1, y1, z1, u1, v1, w1, p1, t1 = loadData(fileName)

    fileName = "Soln_{0:09.4f}.h5".format(outTime)
    x2, y2, z2, u2, v2, w2, p2, t2 = loadData(fileName)

    X1, Y1 = np.meshgrid(x1, y1)
    X2, Y2 = np.meshgrid(x2, y2)
    for n in range(10):
        fig = plt.figure(figsize=(20, 8))

        i = int(t1.shape[2]*n/10)
        ax = fig.add_subplot(1, 2, 1)
        im = ax.contourf(X1, Y1, t1[:, :, i].transpose(), cmap=cm.jet)

        i = int(t2.shape[2]*n/10)
        ax = fig.add_subplot(1, 2, 2)
        im = ax.contourf(X2, Y2, t2[:, :, i].transpose(), cmap=cm.jet)

        #fig.subplots_adjust(right=0.8)
        #cb = fig.colorbar(im)
        #cb.ax.tick_params(labelsize=20)

        plt.savefig(nArray[vNum - 1] + "_Horz_{0:02d}.png".format(n))
        plt.close()

    return 0


def plotVert(vNum):
    global nArray
    global inpTime, outTime

    fileName = "Soln_{0:09.4f}.h5".format(inpTime)
    x1, y1, z1, u1, v1, w1, p1, t1 = loadData(fileName)

    fileName = "Soln_{0:09.4f}.h5".format(outTime)
    x2, y2, z2, u2, v2, w2, p2, t2 = loadData(fileName)

    X1, Z1 = np.meshgrid(x1, z1)
    X2, Z2 = np.meshgrid(x2, z2)
    for n in range(10):
        fig = plt.figure(figsize=(20, 8))

        i = int(t1.shape[1]*n/10)
        ax = fig.add_subplot(1, 2, 1)
        im = ax.contourf(X1, Z1, t1[:, i, :].transpose(), cmap=cm.jet)

        i = int(t2.shape[1]*n/10)
        ax = fig.add_subplot(1, 2, 2)
        im = ax.contourf(X2, Z2, t2[:, i, :].transpose(), cmap=cm.jet)

        plt.savefig(nArray[vNum - 1] + "_Vert_{0:04d}.png".format(i))
        plt.close()

    return 0


#plotHorz(5)
plotVert(5)

