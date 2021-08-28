#!/usr/bin/python

from scipy import interpolate
import numpy as np
import h5py as hp

L = 1.0, 1.0, 1.0

# Input file params
inpTime = 0
nInp = 64, 64, 64
betaInp = 1.2, 1.2, 1.2

# Output file params
outTime = 0
nOut = 128, 128, 128
betaOut = 1.3, 1.3, 1.3

rstTime = 0


def makeGrids():
    global L, xInp, xOut
    global inpTime, nInp, betaInp
    global outTime, nOut, betaOut

    xInp = []
    xOut = []
    if inpTime:
        fileName = "Soln_{0:09.4f}.h5".format(inpTime)

        sFile = hp.File(fileName, 'r')

        xInp.append(np.array(sFile["X"]))
        xInp.append(np.array(sFile["Y"]))
        xInp.append(np.array(sFile["Z"]))

        sFile.close()
    else:
        for i in range(3):
            xi = transGen(nInp[i])

            if betaInp[i]:
                xInp.append(np.array([L[i]*(1.0 - np.tanh(betaInp[i]*(1.0 - 2.0*x))/np.tanh(betaInp[i]))/2.0 for x in xi]))
            else:
                xInp.append(np.copy(xi))

    for i in range(3):
        xi = transGen(nOut[i])

        if betaOut[i]:
            xOut.append(np.array([L[i]*(1.0 - np.tanh(betaOut[i]*(1.0 - 2.0*x))/np.tanh(betaOut[i]))/2.0 for x in xi[1:-1]]))
        else:
            xOut.append(np.copy(xi[1:-1]))


def transGen(N):
    oPts = np.linspace(0.0, 1.0, N+1)
    xi = np.zeros(N + 2)
    xi[1:-1] = (oPts[1:] + oPts[:-1])/2.0
    xi[0] = 0.0
    xi[-1] = 1.0

    return xi


def loadData():
    global inpTime
    global nInp, nOut
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
        rstTime = np.array(f['Time'])

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

    tInp[0, :, :] = tInp[1, :, :]
    tInp[:, 0, :] = tInp[:, 1, :]
    tInp[:, :, 0] = 1.0

    tInp[-1, :, :] = tInp[-2, :, :]
    tInp[:, -1, :] = tInp[:, -2, :]
    tInp[:, :, -1] = 0.0


def xInterp(fInp):
    global xInp, xOut

    lOut, mOut, nOut = fInp.shape
    lOut = Nx

    fOut = np.zeros((lOut, mOut, nOut))

    for j in range(mOut):
        for k in range(nOut):
            intFunct = interpolate.interp1d(xInp[0], fInp[:, j, k], kind='cubic')
            fOut[:, j, k] = intFunct(xOut[0])

    return fOut


def yInterp(fInp):
    global xInp, xOut

    lOut, mOut, nOut = fInp.shape
    mOut = Ny

    fOut = np.zeros((lOut, mOut, nOut))

    for i in range(lOut):
        for k in range(nOut):
            intFunct = interpolate.interp1d(xInp[1], fInp[i, :, k], kind='cubic')
            fOut[i, :, k] = intFunct(xOut[1])

    return fOut


def zInterp(fInp):
    global xInp, xOut

    lOut, mOut, nOut = fInp.shape
    nOut = Nz

    fOut = np.zeros((lOut, mOut, nOut))

    for i in range(lOut):
        for j in range(mOut):
            intFunct = interpolate.interp1d(xInp[2], fInp[i, j, :], kind='cubic')
            fOut[i, j, :] = intFunct(xOut[2])

    return fOut


def interpData():
    global uInp, vInp, wInp, tInp, pInp
    global uOut, vOut, wOut, tOut, pOut

    print("Interpolating Vx")
    uOut = zInterp(yInterp(xInterp(uInp)))

    print("Interpolating Vy")
    vOut = zInterp(yInterp(xInterp(vInp)))

    print("Interpolating Vz")
    wOut = zInterp(yInterp(xInterp(wInp)))

    print("Interpolating P")
    pOut = zInterp(yInterp(xInterp(pInp)))

    print("Interpolating T")
    tOut = zInterp(yInterp(xInterp(tInp)))


def writeFile():
    global restartTime
    global uOut, vOut, wOut, tOut, pOut

    fileName = "newRestart.h5"

    print("Writing into file " + fileName + "\n")

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
    dset = f.create_dataset("Time", data = restartTime)

    f.close()


def main():
    makeGrids()
    loadData()
    interpData()
    writeFile()


main()
