import h5py as hp
import numpy as np

dataDir = "../data/2d_data/7_1e14/"

#bTimes = [199.99, 219.99, 224.99]
bTimes = [199.99, 203.99, 207.99, 211.99, 215.99, 219.99]

getProfs = False

if getProfs:
    uAvg = []
    wAvg = []
    tAvg = []
    tRMS = []
    uRMS = []
    tLst = []
else:
    cTS = []

for i in range(len(bTimes) - 1):
    stTime = bTimes[i]
    enTime = bTimes[i+1]

    ofSuffix = "_{0:06.2f}_{1:06.2f}".format(stTime, enTime)

    if getProfs:
        ofName = "profile_data" + ofSuffix + ".h5"
        sFile = hp.File(dataDir + "output/" + ofName, 'r')

        uAvg.append(np.array(sFile["uAvg"]))
        wAvg.append(np.array(sFile["wAvg"]))
        tAvg.append(np.array(sFile["tAvg"]))
        tRMS.append(np.array(sFile["tRMS"]))
        uRMS.append(np.array(sFile["uRMS"]))
        tLst.append(np.array(sFile["time"]))

        sFile.close()
    else:
        ofName = "rbc_analysis" + ofSuffix + ".dat"
        cTS.append(np.loadtxt(dataDir + "output/" + ofName))

ofSuffix = "_{0:06.2f}_{1:06.2f}".format(bTimes[0], bTimes[-1])

if getProfs:
    uAvg = np.concatenate(uAvg, axis=0)
    wAvg = np.concatenate(wAvg, axis=0)
    tAvg = np.concatenate(tAvg, axis=0)
    tRMS = np.concatenate(tRMS, axis=0)
    uRMS = np.concatenate(uRMS, axis=0)
    tLst = np.concatenate(tLst, axis=0)

    ofName = "profile_data" + ofSuffix + ".h5"
    sFile = hp.File(dataDir + "output/" + ofName, 'w')

    dset = sFile.create_dataset("uAvg", data=uAvg)
    dset = sFile.create_dataset("wAvg", data=wAvg)
    dset = sFile.create_dataset("tAvg", data=tAvg)
    dset = sFile.create_dataset("tRMS", data=tRMS)
    dset = sFile.create_dataset("uRMS", data=uRMS)
    dset = sFile.create_dataset("time", data=tLst)

    sFile.close()
else:
    cTS = np.concatenate(cTS, axis=0)
    ofName = "rbc_analysis" + ofSuffix + ".dat"
    np.savetxt(dataDir + "output/" + ofName, cTS)

