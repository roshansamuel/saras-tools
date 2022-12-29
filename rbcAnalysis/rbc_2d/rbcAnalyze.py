import sys
import h5py as hp
import numpy as np
import derivCalc as df
import nlinCalc as nlin
import diffCalc as diff
import matplotlib as mpl
import globalData as glob
import nusseltCalc as nuss
from globalData import Nx, Nz
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit

mpl.style.use('classic')

# Pyplot-specific directives
plt.rcParams["font.family"] = "serif"

print()

def loadData(fileName):
    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    glob.U = np.pad(np.array(sFile["Vx"]), 1)
    glob.W = np.pad(np.array(sFile["Vz"]), 1)
    glob.P = np.pad(np.array(sFile["P"]), 1)
    glob.T = np.pad(np.array(sFile["T"]), 1)

    glob.X = np.pad(np.array(sFile["X"]), (1, 1), 'constant', constant_values=(0, glob.Lx))
    glob.Z = np.pad(np.array(sFile["Z"]), (1, 1), 'constant', constant_values=(0, glob.Lz))

    sFile.close()

    imposeBCs()


def imposeBCs():
    # Periodic along X
    glob.U[0,:], glob.U[-1,:] = glob.U[-2,:], glob.U[1,:]
    glob.W[0,:], glob.W[-1,:] = glob.W[-2,:], glob.W[1,:]
    glob.P[0,:], glob.P[-1,:] = glob.P[-2,:], glob.P[1,:]
    glob.T[0,:], glob.T[-1,:] = glob.T[-2,:], glob.T[1,:]

    # RBC
    glob.U[:,0], glob.U[:,-1] = -glob.U[:,1], -glob.U[:,-2]
    glob.W[:,0], glob.W[:,-1] = -glob.W[:,1], -glob.W[:,-2]
    glob.P[:,0], glob.P[:,-1] =  glob.P[:,1],  glob.P[:,-2]
    glob.T[:,0], glob.T[:,-1] = 2.0 - glob.T[:,1], -glob.T[:,-2]

    if np.allclose(glob.X[1:-1], glob.xPts[1:-1]):
        glob.X = glob.xPts.copy()
    else:
        print("\tWARNING: Grids are inconsistent along X")

    if np.allclose(glob.Z[1:-1], glob.zPts[1:-1]):
        glob.Z = glob.zPts.copy()
    else:
        print("\tWARNING: Grids are inconsistent along Z")


def getTerms():
    # Compute non-linear terms
    tmpx, tmpz = nlin.computeNLin()
    tmpf = np.sqrt(tmpx**2 + tmpz**2)
    c1 = integrate.simps(integrate.simps(tmpf, glob.Z), glob.X)/glob.tVol

    # Compute pressure terms
    tmpx, tmpz = df.dfx(glob.P), df.dfz(glob.P)
    tmpf = np.sqrt(tmpx**2 + tmpz**2)
    c2 = integrate.simps(integrate.simps(tmpf, glob.Z), glob.X)/glob.tVol

    # Compute buoyancy term
    tmpf = np.abs(glob.T)
    c3 = integrate.simps(integrate.simps(tmpf, glob.Z), glob.X)/glob.tVol

    # Compute diffusion terms
    tmpx, tmpz = diff.computeDiff()
    tmpf = np.sqrt(tmpx**2 + tmpz**2)
    c4 = integrate.simps(integrate.simps(tmpf, glob.Z), glob.X)/glob.tVol

    # Compute dissipation
    tmpx, tmpz = nuss.computeDiss()
    c5 = integrate.simps(integrate.simps(tmpx, glob.Z), glob.X)/glob.tVol
    c6 = integrate.simps(integrate.simps(tmpz, glob.Z), glob.X)/glob.tVol

    c5 *= glob.nu
    c6 *= glob.kappa

    return c1, c2, c3, c4, c5, c6


def main():
    # Set some global variables from CLI arguments
    argList = sys.argv[1:]
    if argList and len(argList) == 2:
        glob.startTime = float(argList[0])
        glob.stopTime = float(argList[1])

    ofSuffix = "_{0:06.2f}_{1:06.2f}".format(glob.startTime, glob.stopTime)

    # Load timelist
    tList = np.loadtxt(glob.dataDir + "output/timeList.dat", comments='#')

    if glob.getProfs:
        uAvg = []
        wAvg = []
        tAvg = []
        tRMS = []
        uRMS = []
        tLst = []
    else:
        cTS = []

    for i in range(tList.shape[0]):
        tVal = tList[i]
        if tVal > glob.startTime and tVal < glob.stopTime:
            fileName = glob.dataDir + "output/Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            if glob.getProfs:
                vxAvg = np.mean(np.abs(glob.U), axis=0)
                vzAvg = np.mean(np.abs(glob.W), axis=0)
                tmAvg = np.mean(glob.T, axis=0)
                tmRMS = np.sqrt(np.mean((glob.T - tmAvg)**2, axis=0))
                vxRMS = np.sqrt(np.mean((glob.U - vxAvg)**2, axis=0))

                uAvg.append(vxAvg)
                wAvg.append(vzAvg)
                tAvg.append(tmAvg)
                tRMS.append(tmRMS)
                uRMS.append(vxRMS)

                tLst.append(tVal)

            else:
                # Subtract mean profile before additional calculations
                if glob.useTheta:
                    glob.T -= (1 - glob.Z)

                c1, c2, c3, c4, c5, c6 = getTerms()

                # Nusselt number from eps_U using Shraiman and Siggia exact relation
                Nu1 = 1.0 + ((glob.Lz**4)/(glob.nu**3))*((glob.Pr**2)/glob.Ra)*c5

                # Nusselt number from eps_T using Shraiman and Siggia exact relation
                Nu2 = c6*(glob.Lz**2)/glob.kappa

                cTS.append([tVal, c1, c2, c3, c4, Nu1, Nu2])

    if glob.getProfs:
        uAvg = np.array(uAvg)
        wAvg = np.array(wAvg)
        tAvg = np.array(tAvg)
        tRMS = np.array(tRMS)
        uRMS = np.array(uRMS)
        tLst = np.array(tLst)

        ofName = "profile_data" + ofSuffix + ".h5"
        sFile = hp.File(glob.dataDir + "output/" + ofName, 'w')

        dset = sFile.create_dataset("uAvg", data=uAvg)
        dset = sFile.create_dataset("wAvg", data=wAvg)
        dset = sFile.create_dataset("tAvg", data=tAvg)
        dset = sFile.create_dataset("tRMS", data=tRMS)
        dset = sFile.create_dataset("uRMS", data=uRMS)
        dset = sFile.create_dataset("time", data=tLst)
        sFile.close()
    else:
        cTS = np.array(cTS)
        ofName = "rbc_analysis" + ofSuffix + ".dat"
        np.savetxt(glob.dataDir + "output/" + ofName, cTS)


main()
