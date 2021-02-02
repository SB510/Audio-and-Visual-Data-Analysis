#Modules
from PIL import Image
import numpy as np


##User Variables
imgName = "Pictures_2/DSC_0069.JPG"
 
 
##Script----------
image = Image.open(imgName)  ##Open image
data = np.asarray(image, dtype="int32" )  ##convert image to array
dataG = data[:,:,1]   ##filter Green channels only
B = dataG.sum(axis=1)  #Sum across Rows
Bmax = np.max(B)   ##find Maximum peak intensity


##Output text file
dataFile = imgName + str(Bmax) + ".txt"
np.savetxt(dataFile, B)
print(Bmax)



