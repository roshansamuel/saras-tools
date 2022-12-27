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

# Debug
import sympy as sp

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

    # Subtract mean profile
    if glob.useTheta:
        glob.T -= (1 - glob.Z)


def imposeBCs():
    # Periodic along X
    glob.X[0], glob.X[-1] = -glob.X[1], glob.Lx + glob.X[1]
    glob.U[0,:], glob.U[-1,:] = glob.U[-2,:], glob.U[1,:]
    glob.W[0,:], glob.W[-1,:] = glob.W[-2,:], glob.W[1,:]
    glob.P[0,:], glob.P[-1,:] = glob.P[-2,:], glob.P[1,:]
    glob.T[0,:], glob.T[-1,:] = glob.T[-2,:], glob.T[1,:]

    # RBC
    glob.P[:,0], glob.P[:,-1] = glob.P[:,1], glob.P[:,-2]
    glob.T[:,0], glob.T[:,-1] = 1.0, 0.0


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

    ofName = "rbc_analysis_{0:06.2f}_{1:06.2f}.dat".format(glob.startTime, glob.stopTime)

    # Load timelist
    tList = np.loadtxt(glob.dataDir + "output/timeList.dat", comments='#')

    # Test function
    fig, ax = plt.subplots(1, 2, figsize=(21,5))

    x, z = sp.symbols('x z')
    fsymp = sp.cos(x)*sp.sin(z) + sp.cos(x)*sp.sin(2*z) + 2*sp.cos(2*x)*sp.sin(3*z)
    #fsymp = sp.sin(5*x) + sp.cos(3*z)
    flmbd = sp.lambdify([x, z], fsymp, "numpy")

    grd2fsymp = sp.diff(sp.diff(fsymp, x), x)# + sp.diff(sp.diff(fsymp, z), z)
    grd2flmbd = sp.lambdify([x, z], grd2fsymp, "numpy")

    X, Z = np.meshgrid(glob.xPts, glob.zPts, indexing='ij')

    f = flmbd(X, Z)
    grd2fAn = grd2flmbd(X, Z)
    im1 = ax[0].contourf(X, Z, grd2fAn)
    plt.colorbar(im1, ax=ax[0])

    grd2fCm = df.d2fx2(f)# + df.d2fz2(f)
    im2 = ax[1].contourf(X[3:-3, 3:-3], Z[3:-3, 3:-3], grd2fCm[3:-3, 3:-3])
    #im2 = ax[1].contourf(X, Z, grd2fCm)
    plt.colorbar(im2, ax=ax[1])

    grdDiff = grd2fAn - grd2fCm
    print(np.max(grdDiff[256:512, 128:256]))
    #exit()

    plt.tight_layout()
    plt.show()
    exit()

    cTS = []
    for i in range(tList.shape[0]):
        tVal = tList[i]
        if tVal > glob.startTime and tVal < glob.stopTime:
            fileName = glob.dataDir + "output/Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            c1, c2, c3, c4, c5, c6 = getTerms()

            # Nusselt number from eps_U using Shraiman and Siggia exact relation
            Nu1 = 1.0 + ((glob.Lz**4)/(glob.nu**3))*((glob.Pr**2)/glob.Ra)*c5

            # Nusselt number from eps_T using Shraiman and Siggia exact relation
            Nu2 = c6*(glob.Lz**2)/glob.kappa

            cTS.append([tVal, c1, c2, c3, c4, Nu1, Nu2])

    cTS = np.array(cTS)
    np.savetxt(glob.dataDir + "output/" + ofName, cTS)


main()
