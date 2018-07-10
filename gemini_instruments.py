
# class InstClass(object):
#
#     def __init__(self, instfile):
#
#         instlines = []
#         with open(instfile, 'r') as f:
#             [instlines.append(line.strip('\n').split('\t')) for line in f]
#             f.close()
#
#         self.date = []
#         self.insts = []
#         self.gmos_fpu = []
#         self.gmos_disp = []
#         self.f2_fpu = []
#
#         for i in range(2,len(instlines)):
#             self.date.append(instlines[i][0])
#             self.insts.append(instlines[i][1].split(','))
#             self.gmos_fpu.append(instlines[i][2])
#             self.gmos_disp.append(instlines[i][3])
#             self.f2_fpu.append(instlines[i][4])

def getinstruments(instfile):
    """
    Read insturment calendar file into a numpy record array.

    Parameters
    ----------
    instfile : str
        Instrument configuration file name

    Returns
    -------
    numpy.rec.array table
    """
    import numpy as np
    instlines = []
    with open(instfile, 'r') as f:
        [instlines.append(line.strip('\n').split('\t')) for line in f]
        f.close()

    return np.rec.array(instlines[2:], names=['date', 'insts', 'gmos_fpu', 'gmos_disp', 'f2_fpu'])

if __name__=='__main__':
    a = getinstruments('gs_inst_schedule.txt')