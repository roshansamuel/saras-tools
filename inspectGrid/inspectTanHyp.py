import numpy as np

nList = 256, 256, 1536
lList = 0.1, 0.1, 1.0
bList = 1.1, 1.1, 1.3
axesList = ['X', 'Y', 'Z']

print("Current grid has total points:", format(nList[0]*nList[1]*nList[2], ',d'))

maxList = np.zeros(3)
minList = np.zeros(3)

for n in range(3):
    vPts = np.linspace(0.0, 1.0, nList[n]+1)
    xi = np.zeros(nList[n] + 2)
    xi[1:-1] = (vPts[1:] + vPts[:-1])/2.0
    xi[0] = xi[1] - (xi[2] - xi[1])
    xi[-1] = xi[-2] + (xi[-2] - xi[-3])

    if bList[n]:
        xGrid = np.array([lList[n]*(1.0 - np.tanh(bList[n]*(1.0 - 2.0*i))/np.tanh(bList[n]))/2.0 for i in xi])
        minList[n] = xGrid[1] - xGrid[0]
        maxList[n] = xGrid[nList[n]//2] - xGrid[nList[n]//2 - 1]

        print("Maximum spacing along " + axesList[n] + " is", maxList[n])
        print("Minimum spacing along " + axesList[n] + " is", minList[n])
    else:
        minList[n] = maxList[n] = lList[n]/nList[n]
        print("Uniform spacing along " + axesList[n] + " is", minList[n])


arList = [maxList[0]/minList[1],
          maxList[0]/minList[2],
          maxList[1]/minList[2],
          maxList[1]/minList[0],
          maxList[2]/minList[0],
          maxList[2]/minList[1]]

print("Current grid has maximum aspect ratio:", max(arList))

