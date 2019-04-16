import numpy as np
from binvox_rw import read_as_3d_array
import os
import pickle

"""
this file takes in the .binvox files and converts them to numpy arrays 
then pickles them for easy later use
"""

# file_type = "" is equal to your unconverted file type. why? line 19
fy = np.array([])
fx = np.array([])
base = "E:/PycharmProjects/untitled/data/"
intbase = 890
endbase = ".binvox"
dirfiles = os.listdir(base)
cou = 0


for x in dirfiles:
    if "binvox" in x:
        if cou > 50: # I had to implement a timer because the variable would be vulnerable to bit overflow
            break # I am not being pretentious I got a memory error
        cou = cou + 1
        path = base + x
        xtrain = read_as_3d_array(open(path, "rb"))
        xtrain = np.array(xtrain.data.astype(int))
        print(xtrain)
        fx = np.append(fx, xtrain)
        if "chair" in x:
            fy = np.append(fy, 0) # chair label will be later decoded via dir from 0
    """
    uncomment if you used binvox to convert files
     but have non converted files
    if "file_type" in x:
        os.remove(x)
        """
    
print(fx)
pickle.dump(fx, open("XChair.pickle", "wb"), protocol=pickle.DEFAULT_PROTOCOL)
pickle.dump(fy, open("YChair.pickle", "wb"), protocol=pickle.DEFAULT_PROTOCOL)
