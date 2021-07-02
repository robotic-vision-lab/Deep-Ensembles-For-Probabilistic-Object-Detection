import numpy as np
from torchVisionTester.Augmenter import *
from PIL import Image
from numpy import random
from glob import glob
from enum import Enum
from torch import from_numpy
from torch import stack as torchStack
#from tensorflow import convert_to_tensor
#from tensorflow import stack as tfStack
from torchVisionTester.Debugger import Debugger
import os
from torchvision import transforms
e,d = exec, Debugger()
e(d.gIS())
class ClassTypes(Enum):
    NUMPY = 1
    PYTORCH_TENSOR = 2
    TENSORFLOW_TENSOR = 3
    PIL_OBJECT = 4
class Datasets(Enum):
    POD_TEST = 1
    POD_VALIDATION = 2
    POD_TRAIN = 3
class DatasetDirectories:
        
    def addPreSuf(self,name,prefix,suffix):
        """
        Params:
            name - string to add prefix and suffix to
            prefix - prefix of output string
            suffix - suffix of output string
        Returns: string with name in middle, prefix and suffix at beginning and 
        end respectively.
        """
        return prefix + name + suffix
    def getPODTestDirectories(self):
        #prefix = './Datasets/test_data/'
        prefix = os.path.join('E:', '/test_data/')

        suffix = '/'
        directoryNames = [self.addPreSuf(str(i).zfill(6),prefix,suffix) for i in range(17)]
        return directoryNames

    def getDirectories(self, dataset):
        """
        Params: 
            dataset - Dataset to load, choose from Datasets Enum
        Returns: list of directories to load for dataset
        """
        if dataset == Datasets.POD_TEST:
            return self.getPODTestDirectories()
class DataLoader:
    def __init__(self, augmentations = None, probabilities = None):
        """
        To use this class you need tensorflow and pytorch installed. If you don't want to install both of those, simply comment out lines 7 or 6
        Params: 
              augmentations - list of augmentations from Augmenter.py
              probabilities - list of probabilities of applying augmentations
        if augmentations != 
        """
        if augmentations != None:
            self.augmenter = Augmenter()
            self.augmentations = augmentations
            self.probabilities = probabilities

    def getFilePathsFromDirectory(self, directory, dataTypes):
        """
        Params: 
            directories - string path to directory containing images
            dataTypes - list of image types to be loaded in string form
        Returns - list of file paths from directory with data types dataTypes

        """
        fileNames = []
        for i in dataTypes:
            fileNames += glob(directory + '*.' + i)

        return fileNames
    def getFilePathsFromDirectories(self, directories, dataTypes):
        """
        Params: 
            directories - list of string paths to directories containing images
            dataTypes - list of  image types to be loaded in string form
        Returns - list of file paths from directories with data types dataTypes

        """
        fileNames = []
        for i in directories:
            fileNames += self.getFilePathsFromDirectory(i,dataTypes)
        return fileNames
    def getArraysFromFileNames(self, fileNames, classType, stack = True, augment = False):
        """
        Params: 
            fileNames - names of all files to be loaded
            classType - type of image to be loaded. Select from ClassTypes
            stack - if true, stack list of arrays or tensors
        Returns - list of arrays of classType from directories with data type dataTypes

        """
        images = [Image.open(filename).convert('RGB') for filename in fileNames] 
        if classType == ClassTypes.PIL_OBJECT:return pilImages
        images = [np.asarray(image) for image in images] 
        
        if augment: images = self.augmenter.augmentBatchMul(images, self.augmentations, self.probabilities)
        if stack : images = np.stack(images)
        if classType == ClassTypes.NUMPY: return images
        if classType == ClassTypes.PYTORCH_TENSOR:
            tensorImages = []
            for image in images:
                transform = transforms.Compose([transforms.ToTensor()])
                if torch.cuda.is_available():
                    tensorIm = transform(image).cuda()
                else:
                    tensorIm = transform(image)
                tensorImages.append(tensorIm)
            
            if stack: tensorImages = torchStack(tensorImages).float()
            return tensorImages
        if classType == ClassTypes.TENSORFLOW_TENSOR:
            tensorImages = [convert_to_tensor(image) for image in images] 

            if stack: tensorImages = tfStack(tensorImages)
            return tensorImages
    def sample(self,mylist):
        """
        Params:
            - mylist - list
        returns: random element from mylist
        """
        if len(mylist) == 1: return mylist[0]
        randomInt = random.randint(0,len(mylist) -1)
        return mylist[randomInt]
    def sampleN(self,mylist, n):
        """
        Params:
            mylist - list
            n - number of random items to sample with replacement
        returns: n random items from mylist
        """
        return [self.sample(mylist) for i in range(n)]
    def getBatchFromDir(self, directories, dataTypes, classType, batch_size, augment = False):
        """
        Params:
            directories - list of string paths to directories containing images
            dataTypes - list of image types to be loaded in string form e.g. 'jpg', 'png', 'jpeg'
            classType - type of image to be loaded. Select from ClassTypes
            batch_size - number of images to sample
        Returns - batch_size images from directories with data type dataTypes in form of classType

        """
        fileNames = self.getFilePathsFromDirectories(directories,dataTypes)
        batchNames = self.sampleN(fileNames, batch_size)
        arrayList = self.getArraysFromFileNames(batchNames,classType, augment)
        return arrayList
    def getBatchFromFileNames(self,fileNames,classType,batch_size, augment = False):
        """
        Params:
            fileNames - list of paths to images to be loaded
            classType - type of image to be loaded. Select from ClassTypes
            batch_size - number of images to sample
        Returns - batch_size images from  fileNames paths in form of classType

        """
        batchNames = self.sampleN(fileNames, batch_size)
        arrayList = self.getArraysFromFileNames(batchNames,classType, augment)
        return arrayList



        

        

            


