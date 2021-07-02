import Augmenter
from PIL import Image
newImage = Image.open('sample.jpg')
import numpy as np
from DataLoader import *
Aug = Augmenter.Augmenter(True, torchMyArray = False)
newA = np.array(newImage)
#for i in Augmenter.Augmentation:
#    randomE = Aug.augment(newA,i, p = .99)
dl = DataLoader()
directories = ['./t1/', './t2/', './t3/']
dataTypes = ['jpg','png']
batch_size = 3
for i in ClassTypes:
    batch = dl.getBatchFromDir(directories,dataTypes,i,batch_size)




