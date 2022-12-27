import derivCalc as df
import globalData as glob

def diffTerm(F):
    return df.d2fx2(F) + df.d2fz2(F)

def computeDiff():
    return diffTerm(glob.U), diffTerm(glob.W)
