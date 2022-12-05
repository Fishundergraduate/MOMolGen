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

    df = pd.read_csv(dataDir+'/log672h_6lu7/data6/present/scores.csv')
    arr = df.values # points include non-pareto-dominate points

    blist = json.load(open(dataDir+'/log672h_6lu7/data6/present/pareto.json','r'))
    brr = np.array(blist['front'])
    myVecFunc = np.vectorize(rewardtoDocking)
    brr[:,0] = myVecFunc(brr[:,0])
    myFig = plt.figure()
    ax = myFig.add_subplot(projection='3d')
    ax.scatter(arr[0:33,0],arr[0:33,1],arr[0:33,2],color='blue',alpha = 0.1)
    ax.scatter(arr[33:66,0],arr[33:66,1],arr[33:66,2],color='blue',alpha = 0.15)
    ax.scatter(arr[66:99,0],arr[66:99,1],arr[66:99,2],color='blue',alpha = 0.2)
    ax.scatter(arr[99:132,0],arr[99:132,1],arr[99:132,2],color='blue',alpha = 0.25)
    ax.scatter(arr[132:165,0],arr[132:165,1],arr[132:165,2],color='blue',alpha = 0.3)
    ax.scatter(arr[165:198,0],arr[165:198,1],arr[165:198,2],color='blue',alpha = 0.3)
    ax.scatter(arr[198:,0],arr[198:,1],arr[198:,2],color='blue',alpha = 0.7)
    #ax.scatter(arr[:,0],arr[:,1],arr[:,2],color='blue',alpha = 0.7)
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