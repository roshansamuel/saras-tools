import derivCalc as df
import globalData as glob

def viscDiss():
    return (2*(df.dfx(glob.U)**2 + df.dfz(glob.W)**2) + (df.dfz(glob.U) + df.dfx(glob.W))**2)

def tempDiss():
    return df.dfx(glob.T)**2 + df.dfz(glob.T)**2

def computeDiss():
    return viscDiss(), tempDiss()
