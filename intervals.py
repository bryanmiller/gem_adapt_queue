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

if __name__=='__main__':
    print('\nTest intervals.py...')
    print('Expected output: indices of first interval of -1\'s from array')

    int_array = np.array([0, 0, 0, 0, -1, -1, -1, -1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, -1, -1, -1, -1], dtype=int)
    print('input:', int_array)

    negones = np.where(int_array==-1)[0][:]
    ii = np.where(intervals(negones)==1)[0][:]
    first = negones[ii]
    print('output: ',first)

    assert((first == np.array([4,5,6,7])).all())
    print('Test successful!')