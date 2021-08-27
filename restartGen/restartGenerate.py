#!/usr/bin/python

from scipy import interpolate
import numpy as np
import h5py as hp

Lx, Ly, Lz = 1.0, 1.0, 1.0

# Input file params
nx, ny, nz = 64, 64, 64
betaInp = 1.2

# Output file params
Nx, Ny, Nz = 128, 128, 128
betaOut = 1.3

def makeGrids():
    global xInp, yInp, zInp
    global xOut, yOut, zOut

    xiInp = transGen(nx)
    etInp = transGen(ny)
    ztInp = transGen(nz)

    xiOut = transGen(Nx)
    etOut = transGen(Ny)
    ztOut = transGen(Nz)

    if betaInp:
        xInp = np.array([Lx*(1.0 - np.tanh(betaInp*(1.0 - 2.0*i))/np.tanh(betaInp))/2.0 for i in xiInp])
        yInp = np.array([Ly*(1.0 - np.tanh(betaInp*(1.0 - 2.0*i))/np.tanh(betaInp))/2.0 for i in etInp])
        zInp = np.array([Lz*(1.0 - np.tanh(betaInp*(1.0 - 2.0*i))/np.tanh(betaInp))/2.0 for i in ztInp])
    else:
        xInp = np.copy(xiInp)
        yInp = np.copy(etInp)
        zInp = np.copy(ztInp)

    if betaOut:
        xOut = np.array([Lx*(1.0 - np.tanh(betaOut*(1.0 - 2.0*i))/np.tanh(betaOut))/2.0 for i in xiOut[1:-1]])
        yOut = np.array([Ly*(1.0 - np.tanh(betaOut*(1.0 - 2.0*i))/np.tanh(betaOut))/2.0 for i in etOut[1:-1]])
        zOut = np.array([Lz*(1.0 - np.tanh(betaOut*(1.0 - 2.0*i))/np.tanh(betaOut))/2.0 for i in ztOut[1:-1]])
    else:
        xOut = np.copy(xiOut[1:-1])
        yOut = np.copy(etOut[1:-1])
        zOut = np.copy(ztOut[1:-1])


def transGen(N):
    oPts = np.linspace(0.0, 1.0, N+1)
    xi = np.zeros(N + 2)
    xi[1:-1] = (oPts[1:] + oPts[:-1])/2.0
    xi[0] = 0.0
    xi[-1] = 1.0

    return xi


def loadData():
    global restartTime
    global uInp, vInp, wInp, tInp, pInp
    global uOut, vOut, wOut, tOut, pOut

    fileName = "restartFile.h5"

    print("Reading file " + fileName + "\n")

    try:
        f = hp.File(fileName, 'r')
    except:
        print("Could not open file " + fileName + "\n")
        exit()

    uInp = np.zeros((nx + 2, ny + 2, nz + 2))
    vInp = np.zeros((nx + 2, ny + 2, nz + 2))
    wInp = np.zeros((nx + 2, ny + 2, nz + 2))
    pInp = np.zeros((nx + 2, ny + 2, nz + 2))
    tInp = np.zeros((nx + 2, ny + 2, nz + 2))

    uInp[1:-1, 1:-1, 1:-1] = np.array(f['Vx'])
    vInp[1:-1, 1:-1, 1:-1] = np.array(f['Vy'])
    wInp[1:-1, 1:-1, 1:-1] = np.array(f['Vz'])
    pInp[1:-1, 1:-1, 1:-1] = np.array(f['P'])
    tInp[1:-1, 1:-1, 1:-1] = np.array(f['T'])
    restartTime = np.array(f['Time'])

    f.close()

    imposeBCs()

    uOut = np.zeros((Nx, Ny, Nz))
    vOut = np.zeros((Nx, Ny, Nz))
    wOut = np.zeros((Nx, Ny, Nz))
    pOut = np.zeros((Nx, Ny, Nz))
    tOut = np.zeros((Nx, Ny, Nz))


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
            intFunct = interpolate.interp1d(xInp, fInp[:, j, k], kind='cubic')
            fOut[:, j, k] = intFunct(xOut)

    return fOut


def yInterp(fInp):
    global yInp, yOut

    lOut, mOut, nOut = fInp.shape
    mOut = Ny

    fOut = np.zeros((lOut, mOut, nOut))

    for i in range(lOut):
        for k in range(nOut):
            intFunct = interpolate.interp1d(yInp, fInp[i, :, k], kind='cubic')
            fOut[i, :, k] = intFunct(yOut)

    return fOut


def zInterp(fInp):
    global zInp, zOut

    lOut, mOut, nOut = fInp.shape
    nOut = Nz

    fOut = np.zeros((lOut, mOut, nOut))

    for i in range(lOut):
        for j in range(mOut):
            intFunct = interpolate.interp1d(zInp, fInp[i, j, :], kind='cubic')
            fOut[i, j, :] = intFunct(zOut)

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
