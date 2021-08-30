#!/usr/bin/python

from datetime import datetime
from scipy import interpolate
import multiprocessing as mp
import numpy as np
import h5py as hp

L = 1.0, 0.25, 1.0

# Input file params
inpTime = 675
nInp = 64, 64, 64
betaInp = 1.2, 1.2, 1.2

# Output file params
outTime = 320
nOut = 256, 128, 256
betaOut = 1.3, 1.3, 1.3


def makeGrids():
    global inpTime
    global L, xInp, xOut
    global nOut, betaOut, nInp, betaInp

    xInp = []
    xOut = []
    if inpTime:
        fileName = "Soln_{0:09.4f}.h5".format(inpTime)

        sFile = hp.File(fileName, 'r')

        gridAxes = ["X", "Y", "Z"]
        nTmp = list(nInp)
        for i in range(3):
            gData = np.array(sFile[gridAxes[i]])
            nTmp[i] = gData.shape[0]

            gData = np.pad(gData, 1)
            gData[-1] = L[i]

            if gData[-1] - gData[-2] > 5*(gData[-2] - gData[-3]):
                print("WARNING: There could be a domain length mismatch along axis " + gridAxes[i])

            xInp.append(gData)

        nInp = tuple(nTmp)
        sFile.close()
    else:
        for i in range(3):
            xi = transGen(nInp[i])

            if betaInp[i]:
                xInp.append(np.array([L[i]*(1.0 - np.tanh(betaInp[i]*(1.0 - 2.0*x))/np.tanh(betaInp[i]))/2.0 for x in xi]))
            else:
                xInp.append(L[i]*xi)

    for i in range(3):
        xi = transGen(nOut[i])

        if betaOut[i]:
            xOut.append(np.array([L[i]*(1.0 - np.tanh(betaOut[i]*(1.0 - 2.0*x))/np.tanh(betaOut[i]))/2.0 for x in xi[1:-1]]))
        else:
            xOut.append(L[i]*xi[1:-1])


def transGen(N):
    oPts = np.linspace(0.0, 1.0, N+1)
    xi = np.zeros(N + 2)
    xi[1:-1] = (oPts[1:] + oPts[:-1])/2.0
    xi[0] = 0.0
    xi[-1] = 1.0

    return xi


def loadData():
    global nInp, nOut
    global inpTime, outTime
    global uInp, vInp, wInp, tInp, pInp
    global uOut, vOut, wOut, tOut, pOut

    if inpTime:
        fileName = "Soln_{0:09.4f}.h5".format(inpTime)
    else:
        fileName = "restartFile.h5"

    print("Reading file " + fileName + "\n")

    try:
        f = hp.File(fileName, 'r')
    except:
        print("Could not open file " + fileName + "\n")
        exit()

    uInp = np.zeros(tuple([nInp[i] + 2 for i in range(3)]))
    vInp = np.zeros(tuple([nInp[i] + 2 for i in range(3)]))
    wInp = np.zeros(tuple([nInp[i] + 2 for i in range(3)]))
    pInp = np.zeros(tuple([nInp[i] + 2 for i in range(3)]))
    tInp = np.zeros(tuple([nInp[i] + 2 for i in range(3)]))

    uInp[1:-1, 1:-1, 1:-1] = np.array(f['Vx'])
    vInp[1:-1, 1:-1, 1:-1] = np.array(f['Vy'])
    wInp[1:-1, 1:-1, 1:-1] = np.array(f['Vz'])
    pInp[1:-1, 1:-1, 1:-1] = np.array(f['P'])
    tInp[1:-1, 1:-1, 1:-1] = np.array(f['T'])

    if not outTime:
        outTime = np.array(f['Time'])

    f.close()

    imposeBCs()

    uOut = np.zeros(nOut)
    vOut = np.zeros(nOut)
    wOut = np.zeros(nOut)
    pOut = np.zeros(nOut)
    tOut = np.zeros(nOut)


def imposeBCs():
    global tInp, pInp

    pInp[0, :, :] = pInp[1, :, :]
    pInp[:, 0, :] = pInp[:, 1, :]
    pInp[:, :, 0] = pInp[:, :, 1]

    pInp[-1, :, :] = pInp[-2, :, :]
    pInp[:, -1, :] = pInp[:, -2, :]
    pInp[:, :, -1] = pInp[:, :, -2]

    # WARNING: Below BCs are for RBC only.
    # Stably stratified flows may require changes to below code.
    tInp[0, :, :] = tInp[1, :, :]
    tInp[:, 0, :] = tInp[:, 1, :]
    tInp[:, :, 0] = 1.0

    tInp[-1, :, :] = tInp[-2, :, :]
    tInp[:, -1, :] = tInp[:, -2, :]
    tInp[:, :, -1] = 0.0


def xInterp(fInp):
    global nOut
    global xInp, xOut

    l, m, n = fInp.shape
    l = nOut[0]

    fOut = np.zeros((l, m, n))

    for j in range(m):
        for k in range(n):
            intFunct = interpolate.interp1d(xInp[0], fInp[:, j, k], kind='cubic')
            fOut[:, j, k] = intFunct(xOut[0])

    return fOut


def yInterp(fInp):
    global nOut
    global xInp, xOut

    l, m, n = fInp.shape
    m = nOut[1]

    fOut = np.zeros((l, m, n))

    for i in range(l):
        for k in range(n):
            intFunct = interpolate.interp1d(xInp[1], fInp[i, :, k], kind='cubic')
            fOut[i, :, k] = intFunct(xOut[1])

    return fOut


def zInterp(fInp):
    global nOut
    global xInp, xOut

    l, m, n = fInp.shape
    n = nOut[2]

    fOut = np.zeros((l, m, n))

    for i in range(l):
        for j in range(m):
            intFunct = interpolate.interp1d(xInp[2], fInp[i, j, :], kind='cubic')
            fOut[i, j, :] = intFunct(xOut[2])

    return fOut


def interpData(interpVar):
    global uInp, vInp, wInp, tInp, pInp

    if interpVar == 0:
        print("Interpolating Vx")
        return xInterp(yInterp(zInterp(uInp)))

    elif interpVar == 1:
        print("Interpolating Vy")
        return xInterp(yInterp(zInterp(vInp)))

    elif interpVar == 2:
        print("Interpolating Vz")
        return xInterp(yInterp(zInterp(wInp)))

    elif interpVar == 3:
        print("Interpolating P")
        return xInterp(yInterp(zInterp(pInp)))

    elif interpVar == 4:
        print("Interpolating T")
        return xInterp(yInterp(zInterp(tInp)))


def writeFile():
    global xOut
    global outTime
    global uOut, vOut, wOut, tOut, pOut

    fileName = "Soln_{0:09.4f}.h5".format(outTime)

    print("\nWriting into file " + fileName)

    try:
        f = hp.File(fileName, 'w')
    except:
        print("Could not open file " + fileName + "\n")
        exit()

    dset = f.create_dataset("Vx", data = uOut)
    dset = f.create_dataset("Vy", data = vOut)
    dset = f.create_dataset("Vz", data = wOut)
    dset = f.create_dataset("P", data = pOut)
    dset = f.create_dataset("T", data = tOut)

    dset = f.create_dataset("X", data = xOut[0])
    dset = f.create_dataset("Y", data = xOut[1])
    dset = f.create_dataset("Z", data = xOut[2])

    dset = f.create_dataset("Time", data = outTime)

    f.close()


def main():
    global uOut, vOut, wOut, tOut, pOut

    makeGrids()
    loadData()

    # Lazy parallelism
    nProcs = 5
    pool = mp.Pool(processes=nProcs)

    t1 = datetime.now()
    poolRes = [pool.apply_async(interpData, args=(x,)) for x in range(nProcs)]
    uOut, vOut, wOut, pOut, tOut = [x.get() for x in poolRes]
    t2 = datetime.now()

    writeFile()

    print("\nTime taken for interpolation = ", t2 - t1)


main()
