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

a = -zStag[0]
b = zStag[1]
l = a + b
cwal = l/b
cvar = a/b
print(cwal, cvar)
'''
c = zStag[3] - zStag[2]
b = zStag[2] - zStag[1]
a = b*b/c
zStag[0] = zStag[1] - a
print(zStag[:4])
print(b, c, a)
'''

minLength = zStag[1] - zStag[0]
maxLength = zStag[Nz//2] - zStag[Nz//2 - 1]
aspectRat = maxLength/minLength

print("Current grid has minimum spacing of", minLength)
print("Current grid has maximum spacing of", maxLength)
print("Current grid has aspect ratio of", aspectRat)

zDiff = zStag[1:] - zStag[:-1]
#plt.plot(zDiff)
#plt.show()
