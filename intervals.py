import numpy as np

def intervals(int_array):

    # find the number and properties of contiguous intervals in a
    # vector of indices

    ni = len(int_array)
    cvec = np.zeros(ni,dtype=int)
    nint=1
    cvec[0] = nint
    for j in range(1,ni):
        if (int_array[j] != (int_array[j-1] + 1)):
            nint=nint+1
        cvec[j] = nint

    indx = np.digitize(cvec,bins=np.arange(ni)+1)
    
    return indx
