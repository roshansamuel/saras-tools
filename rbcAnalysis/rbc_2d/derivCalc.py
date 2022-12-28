import numpy as np
from globalData import Nx, Nz, xi_x, zt_z, xixx, ztzz, xix2, ztz2, i2hx, i2hz, ihx2, ihz2

xSt, xEn = 1, Nx + 1
x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

zSt, zEn = 1, Nz + 1
z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

# First derivative with 2nd order central difference
def dfx(F):
    tmp = np.zeros_like(F)
    tmp[x0, :] = (F[xp1, :] - F[xm1, :]) * xi_x[x0]
    tmp[ 0, :] = (-3.0*F[ 0, :] + 4*F[ 1, :] - F[ 2, :]) * xi_x[ 0]
    tmp[-1, :] = ( 3.0*F[-1, :] - 4*F[-2, :] + F[-3, :]) * xi_x[-1]

    return tmp * i2hx

def dfz(F):
    tmp = np.zeros_like(F)
    tmp[:, z0] = (F[:, zp1] - F[:, zm1]) * zt_z[z0]
    tmp[:,  0] = (-3.0*F[:,  0] + 4*F[:,  1] - F[:,  2]) * zt_z[ 0]
    tmp[:, -1] = ( 3.0*F[:, -1] - 4*F[:, -2] + F[:, -3]) * zt_z[-1]

    return tmp * i2hz


# Second derivative with 2nd order central difference
def d2fx2(F):
    tmp = np.zeros_like(F)
    tmp[x0, :] = ((F[xp1, :] + F[xm1, :] - 2.0*F[x0, :]) * ihx2 * xix2[x0]) + ((F[xp1, :] - F[xm1, :]) * i2hx * xixx[x0])
    tmp[ 0, :] = ((F[ 2, :] + F[ 0, :] - 2.0*F[ 1, :]) * ihx2 * xix2[ 0]) + ((-3.0*F[ 0, :] + 4*F[ 1, :] - F[ 2, :]) * i2hx * xixx[ 0])
    tmp[-1, :] = ((F[-3, :] + F[-1, :] - 2.0*F[-2, :]) * ihx2 * xix2[-1]) + (( 3.0*F[-1, :] - 4*F[-2, :] + F[-3, :]) * i2hx * xixx[-1])

    return tmp

def d2fz2(F):
    tmp = np.zeros_like(F)
    tmp[:, z0] = ((F[:, zp1] + F[:, zm1] - 2.0*F[:, z0]) * ihz2 * ztz2[z0]) + ((F[:, zp1] - F[:, zm1]) * i2hz * ztzz[z0])
    tmp[:,  0] = ((F[:,  2] + F[:,  0] - 2.0*F[:,  1]) * ihz2 * ztz2[ 0]) + ((-3.0*F[:,  0] + 4*F[:,  1] - F[:,  2]) * i2hz * ztzz[ 0])
    tmp[:, -1] = ((F[:, -3] + F[:, -1] - 2.0*F[:, -2]) * ihz2 * ztz2[-1]) + (( 3.0*F[:, -1] - 4*F[:, -2] + F[:, -3]) * i2hz * ztzz[-1])

    return tmp
