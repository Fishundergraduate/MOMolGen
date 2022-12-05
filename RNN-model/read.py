import h5py
import os

f = h5py.File(os.path.dirname(__file__)+'/model.h5','r')
for key in f.keys():
    print(f[key].name)
    
    #print(f[key].value)
    #print(f[key].head)