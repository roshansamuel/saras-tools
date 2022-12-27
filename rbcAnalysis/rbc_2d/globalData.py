import numpy as np
import yamlRead as yr

dataDir = "../data/2d_data/2_Ra_1e9_pySaras/"

# Is YAML file available?
readYAML = True

# Global variables
Lx, Lz = 1.0, 1.0
Nx, Nz = 100, 100
btX, btZ = 0.0, 0.0
U, W, P, T, X, Z = 1, 1, 1, 1, 1, 1

# Simulation parameters
Pr = 1.0
Ra = 1e1
nu, kappa = 1.0, 1.0

# Use theta (temperature fluctuation) instead of T (temperature)
useTheta = True

# Read existing data vs compute anew
readFile = False

# Specify starting time for calculations. Use 0 to include all in timeList.dat
startTime = 0.0

# Specify starting time for calculations. Use Inf to include all in timeList.dat
stopTime = float('Inf')

# If YAML file is available, parse it
if readYAML:
    yr.parseYAML(dataDir)

yr.updateDissCoeffs()

tVol = Lx*Lz

dXi = 1.0/(Nx)
dZt = 1.0/(Nz)

i2hx = 1.0/(2.0*dXi)
i2hz = 1.0/(2.0*dZt)

ihx2 = 1.0/(dXi**2.0)
ihz2 = 1.0/(dZt**2.0)

def genGrid(N, L, bt):
    vPts = np.linspace(0.0, 1.0, N+1)
    xi = np.pad((vPts[1:] + vPts[:-1])/2.0, (1, 1), 'constant', constant_values=(0.0, 1.0))
    if bt:
        xPts = np.array([L*(1.0 - np.tanh(bt*(1.0 - 2.0*i))/np.tanh(bt))/2.0 for i in xi])
        xi_x = np.array([np.tanh(bt)/(L*bt*(1.0 - ((1.0 - 2.0*k/L)*np.tanh(bt))**2.0)) for k in xPts])
        xixx = np.array([-4.0*(np.tanh(bt)**3.0)*(1.0 - 2.0*k/L)/(L**2*bt*(1.0 - (np.tanh(bt)*(1.0 - 2.0*k/L))**2.0)**2.0) for k in xPts])
        xix2 = np.array([k*k for k in xi_x])
    else:
        xPts = L*xi
        xi_x = np.ones_like(xPts)/L
        xix2 = xi_x**2
        xixx = np.zeros_like(xPts)

    return xi, xPts, xi_x, xixx, xix2

xi, xPts, xi_x, xixx, xix2 = genGrid(Nx, Lx, btX)
zt, zPts, zt_z, ztzz, ztz2 = genGrid(Nz, Lz, btZ)

npax = np.newaxis
xi_x, xixx, xix2 = xi_x[:, npax], xixx[:, npax], xix2[:, npax]
