import numpy as np

def timingwindows(time_consts, t1, t2):
    """
    Compute all time constraint windows for a Gemini observation within a set time frame.

    Return a list of time pairs or 'NoneType' if no valid windows exist within the time frame.

    Parameters
    ----------
    time_const : list of dictionaries
        Gemini observation time constraints (eg. time_consts = [{'start': '~astropy.time.Time',
            'duration': '~astropy.units.Quantity', 'repeat': int, 'period': '~astropy.time.Time'},...])

    t1 : '~astropy.time.Time'
        start time of scheduling period

    t2 : '~astropy.time.Time'
        end time of scheduling period

    Returns
    -------
    array of time windows : list of '~astropy.time.Time' pairs

    None : if there are no valid time windows
    """

    windows = []

    if time_consts is None:  # No time constraint
        windows.append([t1, t2])
    else:
        for time_const in time_consts: # Cycle through time constraints for observation
            n_reps = time_const['repeat'] + 1

            for i in range(n_reps): # Cycle through repeats in time constraint

                # Set start and end times of current observable time window
                if time_const['duration'] == -1.0:
                    wstart = time_const['start']
                    wend = t2
                else:
                    wstart = time_const['start'] + time_const['period'] * i
                    wend = wstart + time_const['duration']

                # trim window to be within start and end of night
                if (t1 < wend) and (wstart < t2):
                    wstart = np.max([wstart, t1])
                    wend = np.min([wend, t2])
                    windows.append([wstart, wend])
                elif wstart>=t2:
                    break
                else:
                    pass

    if len(windows)==0:
        windows = None
    return windows