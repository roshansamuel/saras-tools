#import matplotlib.pyplot as plt
import numpy as np

Nz = 1024
zLen = 1.0
zBeta = 1.3

vPts = np.linspace(0.0, 1.0, Nz+1)
zt = np.zeros(Nz + 2)
zt[1:-1] = (vPts[1:] + vPts[:-1])/2.0
zt[0] = zt[1] - (zt[2] - zt[1])
zt[-1] = zt[-2] + (zt[-2] - zt[-3])
zStag = np.array([zLen*(1.0 - np.tanh(zBeta*(1.0 - 2.0*i))/np.tanh(zBeta))/2.0 for i in zt])

print(zStag[:6])

minLength = zStag[1] - zStag[0]
maxLength = zStag[Nz//2] - zStag[Nz//2 - 1]
aspectRat = maxLength/minLength

print("Current grid has minimum spacing of", minLength)
print("Current grid has maximum spacing of", maxLength)
print("Current grid has aspect ratio of", aspectRat)

zDiff = zStag[1:] - zStag[:-1]
#plt.plot(zDiff)
#plt.show()
