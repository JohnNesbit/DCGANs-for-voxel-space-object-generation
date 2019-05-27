import numpy as np
from binvox_rw import read_as_3d_array
import os
import pickle
import re
import time

"""
this file takes in the .binvox files and converts them to numpy arrays 
then pickles them for easy later use
"""

fy = np.array([])
fx = np.array([])


fdir = {
    0: "bathtub",
    1: "bed",
    2:"chairs",
    3: "desk",
    4: "dresser",
    5: "monitor",
    6: "night_stand",
    7: "sofa",
    8: "table",
    9: "toilet"
}

ifdir = {
    "bathtub": 1,
    "bed": 2,
    "chairs": 3,
    "desk": 4,
    "dresser": 5,
    "monitor": 6,
    "night_stand": 7,
    "sofa": 8,
    "table": 9,
    "toilet": 10
}


base = "E:/PycharmProjects/untitled/%s/"


intbase = 890
endbase = ".binvox"



xc = np.array([])
yc = np.array([])
for _ in range(9):
    for i in os.listdir(base % fdir[_]):
        for x in os.listdir(base % fdir[_] + i):
            print(x)
            path = base % fdir[_] + i + "/" + x
            if "off" in x:
                os.system(path)
                time.sleep(1)
                x = re.sub(r".off", ".binvox",  str(x))
                path = base % fdir[_] + i + "/" + x
                xtrain = read_as_3d_array(open(path, "rb"))
                xtrain = np.array(xtrain.data.astype(int))
                print(xtrain.shape)
                xc = np.append(fx, xtrain)
                yc = np.append(fy, _)  # chair label will be later decoded via dir from 0
            if "binvox" in x:
                xtrain = read_as_3d_array(open(path, "rb"))
                xtrain = np.array(xtrain.data.astype(int))
                print(xtrain.shape)
                xc = np.append(fx, xtrain)
                yc = np.append(fy, _) # chair label will be later decoded via dir from 0

print(fx.shape)
pickle.dump(fx, open("Xdata.pickle", "wb"), protocol=pickle.DEFAULT_PROTOCOL)
pickle.dump(fy, open("Ydata.pickle", "wb"), protocol=pickle.DEFAULT_PROTOCOL)
