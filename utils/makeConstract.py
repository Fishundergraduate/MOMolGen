import argparse
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np

def rewardtoDocking(r: float)->float:
    return 10*r/(r-1)

def getMax(a: float , b: float ) -> float:
    return a if a> b else b

def getMin(a: float , b: float ) -> float:
    return a if a< b else b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir',type=str)
    args = parser.parse_args()
    dataDir = args.dataDir

    #df1 = pd.read_csv(dataDir+'/diff0.csv')
    df1 = pd.read_csv(dataDir+'/diff24.csv')
    df10 = pd.read_csv(dataDir+'/diff48.csv')
    df11 = pd.read_csv(dataDir+'/diff72.csv')
    df12 = pd.read_csv(dataDir+'/diff96.csv')
    df13 = pd.read_csv(dataDir+'/diff120.csv')
    df14 = pd.read_csv(dataDir+'/diff144.csv')
    df2 = pd.read_csv(dataDir+'/diff168.csv')
    df3 = pd.read_csv(dataDir+'/diff336.csv')
    arr1 = df1.values # points include non-pareto-dominate points
    arr10 = df10.values # points include non-pareto-dominate points
    arr11 = df11.values # points include non-pareto-dominate points
    arr12 = df12.values # points include non-pareto-dominate points
    arr13 = df13.values # points include non-pareto-dominate points
    arr14 = df14.values # points include non-pareto-dominate points
    arr2 = df2.values # points include non-pareto-dominate points
    arr3 = df3.values # points include non-pareto-dominate points

    blist = json.load(open(dataDir+'/log504h/data3'+'/present/pareto.json','r'))
    brr = np.array(blist['front'])
    myVecFunc = np.vectorize(rewardtoDocking)
    brr[:,0] = myVecFunc(brr[:,0])
    myFig = plt.figure()
    ax = myFig.add_subplot(projection='3d')
    ax.scatter(arr1[:,0],arr1[:,1],arr1[:,2],color='blue',alpha = 0.1)
    ax.scatter(arr10[:,0],arr10[:,1],arr10[:,2],color='blue',alpha = 0.15)
    ax.scatter(arr11[:,0],arr11[:,1],arr11[:,2],color='blue',alpha = 0.2)
    ax.scatter(arr12[:,0],arr12[:,1],arr12[:,2],color='blue',alpha = 0.25)
    ax.scatter(arr13[:,0],arr13[:,1],arr13[:,2],color='blue',alpha = 0.3)
    ax.scatter(arr14[:,0],arr14[:,1],arr14[:,2],color='blue',alpha = 0.7)
    ax.scatter(arr2[:,0],arr2[:,1],arr2[:,2],color='blue',alpha = 0.8)
    ax.scatter(arr3[:,0],arr3[:,1],arr3[:,2],color='blue',alpha = 0.9)
    ax.scatter(brr[:,0],brr[:,1],brr[:,2],color='red',alpha = 1.0)

    
    #ax.set_xlim3d(min(np.concatenate([arr[:,0], brr[:,0]])), max(np.concatenate([arr[:,0], brr[:,0]])))#Docking
    ax.set_xlim3d(min(brr[:,0]),-5)#Docking
    #ax.set_ylim3d(min(np.concatenate([arr[:,1], brr[:,1]])), max(np.concatenate([arr[:,1], brr[:,1]])))#QED
    ax.set_ylim3d(0, 1)#QED
    #ax.set_zlim3d(min(np.concatenate([arr[:,2], brr[:,2]])), max(np.concatenate([arr[:,2], brr[:,2]])))#Toxicity
    ax.set_zlim3d(0, max(brr[:,2]))#Toxicity
    
    ax.set_xlabel("Docking Score")
    ax.set_ylabel("QED score")
    ax.set_zlabel("Toxicity Probability")
    myFig.savefig(f'{dataDir}/plotfig.png')