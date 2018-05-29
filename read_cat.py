import numpy as np
import re

__all__ = ["create"]

class Catalog(object):

    """
    Rough version of a container class for parsing catalog and storing information.
    
    Attribute naming convention: lowercase column headers w/ underscores (eg.'Obs. Status'=obs_status).

    Issue: repeated column names are not combined into a single attribute name.
    If an column name is repeated, a numerical value is appended to the corresponding attr. name (eg. 'disperser_2').
    """

    def __init__(self,otfile=None):

        print('\nReading '+otfile+'...')
        cattext = []
        with open(otfile, 'r') as readcattext: #read file into memory. 
            [cattext.append(line.split('\t')) for line in readcattext] #Split lines where tabs ('\t') are found.
            readcattext.close()

        names = np.array(cattext[8])
        obsall = np.array(cattext[10:])

        print('\notcat attribute names...')
        existing_names = []
        for i in range(0,len(names)):
            #remove special charaters, trim ends of string, replace whitespace with underscore
            string=names[i].replace('.','')
            string=re.sub(r'\W',' ',string)
            string=string.strip()
            string=re.sub(r' +','_',string)
            string=string.lower()
            if string=='class': #change attribute name (python doesn't allow attribute name 'class')
                string = 'obs_class'
            if np.isin(string,existing_names): #add number to end of repeated attribute name
                rename = True
                j = 0    
                while rename:
                    if j>=8: #if number 9 is reached, make next number 10
                        tempstring = string+'_1'+chr(50+j-10)
                    else:
                        tempstring = string+'_'+chr(50+j)
                    if np.isin(tempstring,existing_names): #if name taken, increment number and check again
                        j+=1
                    else: 
                        string = tempstring
                        rename = False

            setattr(self, string, obsall[:,i]) #set attribute name an include corresponding catalog column
            existing_names.append(string) #add name to library of used names
            #print(string)

        print('\nFound '+str(len(obsall))+' observations in '+str(otfile))

