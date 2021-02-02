#This programis given a list of files and it outputs a mtrix of each file's FFT 

# Load libraries
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import glob
from matplotlib import pyplot
from numpy import interp
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from pandas import read_csv
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from os import listdir


#perform FFT on gven file
def transform(file):
    fs, data = wavfile.read(file) # load the data
    a = data.T # this is a two channel soundtrack, I get the first track
    normalized = []
    for ele in a:
        b=(ele/2**8.)*2-1# this is 8-bit track, b is now normalized on [-1,1)
        normalized.append(b)


    c = fft(normalized) # calculate fourier transform (complex numbers list)
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)
    d = int(d)
    transformed = abs(c[:(d)])
##    print(transformed)
    return transformed, fs, data

#Load files
url = "classifyOutput.csv"
names = ["file", "voiceState"]
dataset = read_csv(url, header=None, names=names)

#Set the frequency domain the all files will be
timeLength = 5 #Seconds
frameRate = 1/44100
fLow = 1/timeLength
fHigh = 4000
fZero = []
i = fLow
while i <= fHigh:
    fZero.append(i)   
    i += fLow
    
#each Audio file to FFT and Interpolate
text = []
num = 0
wavlist = dataset["file"]
with open("data2.csv", "a") as my_csv:
    for file in wavlist: #get the atcual file number
    
        transformed, fs, data = transform('FSD50K.eval_audio/'+str(file) + '.wav')

        
        t = len(data)/fs
        
        fLow = 1/t
        fHigh = 1/(2*(1/fs))

            
        xaxis= []
        i = fLow
        transformed = transformed[:int(4000/fLow)+1]
        while len(xaxis) < len(transformed):
            xaxis.append(i)   
            i += fLow
        
        interpolate = interp(fZero, xaxis, transformed)
             
        write = csv.writer(my_csv, delimiter=',')
        interpolate = np.append(interpolate, dataset['voiceState'][num])
        write.writerow(interpolate)
        print(file)
        num+=1
    

 
