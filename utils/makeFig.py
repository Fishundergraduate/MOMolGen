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

    df = pd.read_csv(dataDir+'/present/scores.csv')
    arr = df.values # points include non-pareto-dominate points

    blist = json.load(open(dataDir+'/present/pareto.json','r'))
    brr = np.array(blist['front'])
    myVecFunc = np.vectorize(rewardtoDocking)
    brr[:,0] = myVecFunc(brr[:,0])
    myFig = plt.figure()
    ax = myFig.add_subplot(projection='3d')
    ax.scatter(arr[0:150,0],arr[0:150,1],arr[0:150,2],color='blue',alpha = 0.2)
    ax.scatter(arr[150:300,0],arr[150:300,1],arr[150:300,2],color='blue',alpha = 0.4)
    ax.scatter(arr[300:,0],arr[300:,1],arr[300:,2],color='blue',alpha = 0.7)
    
    ax.scatter(brr[:,0],brr[:,1],brr[:,2],color='red',alpha = 1.0)

    
    #ax.set_xlim3d(min(np.concatenate([arr[:,0], brr[:,0]])), max(np.concatenate([arr[:,0], brr[:,0]])))#Docking
    ax.set_xlim3d(min(brr[:,0]),-3)#Docking
    #ax.set_ylim3d(min(np.concatenate([arr[:,1], brr[:,1]])), max(np.concatenate([arr[:,1], brr[:,1]])))#QED
    ax.set_ylim3d(0, 1)#QED
    #ax.set_zlim3d(min(np.concatenate([arr[:,2], brr[:,2]])), max(np.concatenate([arr[:,2], brr[:,2]])))#Toxicity
    ax.set_zlim3d(0, max(brr[:,2]))#Toxicity
    
    ax.set_xlabel("Docking Score")
    ax.set_ylabel("QED score")
    ax.set_zlabel("Toxicity Probability")
    myFig.savefig(f'{dataDir}/plotfig.png')