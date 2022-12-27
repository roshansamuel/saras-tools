import yaml as yl
import numpy as np
import globalData as glob

def parseYAML(dataDir):
    paraFile = dataDir + "input/parameters.yaml"
    yamlFile = open(paraFile, 'r')
    try:
        yamlData = yl.load(yamlFile, yl.FullLoader)
    except:
        yamlData = yl.load(yamlFile)

    glob.Lx = float(yamlData["Program"]["X Length"])
    glob.Lz = float(yamlData["Program"]["Z Length"])

    glob.Nx = int(yamlData["Mesh"]["X Size"])
    glob.Nz = int(yamlData["Mesh"]["Z Size"])

    gridType = str(yamlData["Mesh"]["Mesh Type"])

    if gridType[0] == 'U':
        glob.btX = 0.0
    else:
        glob.btX = float(yamlData["Mesh"]["X Beta"])

    if gridType[2] == 'U':
        glob.btZ = 0.0
    else:
        glob.btZ = float(yamlData["Mesh"]["Z Beta"])

    glob.Pr = float(yamlData["Program"]["Prandtl Number"])
    glob.Ra = float(yamlData["Program"]["Rayleigh Number"])

    updateDissCoeffs()

def updateDissCoeffs():
    glob.kappa = 1.0/np.sqrt(glob.Ra*glob.Pr)
    glob.nu = np.sqrt(glob.Pr/glob.Ra)
