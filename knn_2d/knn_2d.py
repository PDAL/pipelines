#/usr/bin/python

# knn_2d.py
### compute the 2 dimensional number of neighbors

import numpy as np
import scipy.spatial as spatial

MAXNEIGHBORS = 250
MAXDIST = 2.0

def compute2dDensity(outs, ins, itm, kd_2d, data_2d):

    maxDist = MAXDIST
    numNbrs = MAXNEIGHBORS

    pt = data_2d[itm]
    nbrs = kd_2d.query(pt, k=numNbrs, distance_upper_bound=maxDist)
    npts = data_2d[nbrs[1][0]]
    rN = [p for p in nbrs[0] if p < np.inf]

    outs['Density2D'][itm] = len(rN)
  

# main routine to call from PDAL
def knn_2d(ins, outs):
    # initialize the outs array
    outs['Density2D'] = np.zeros(ins['X'].shape, dtype=np.float)

    #setup 2d search
    data_2d = np.dstack((ins['X'], ins['Y']))[0]
    kd_2d = spatial.KDTree(data_2d, leafsize=10)

    print(len(ins['X']))
    for itm in range(len(ins['X'])):
        compute2dDensity(outs, ins, itm, kd_2d, data_2d)

    print("All Done")

    return True

if __name__ == "__main__":
    print("this must be called from a PDAL piepline")
