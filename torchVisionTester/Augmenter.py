from PIL import Image, ImageOps
from enum import Enum
import numpy as np
from numpy import random as r
import cv2
import math
from matplotlib import pyplot as plt
import random
import torch
import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
class Augmentation(Enum):
    AGCWD = 1
    RANDOM_ERASE = 2
    FLIP_HORIZONTAL = 3
    RANDOM_CROP = 4
    ALB_BLUR = 5
    ALB_VERTICAL_FLIP = 6
    ALB_HORIZFLIP = 7
    ALB_FLIP = 8
    ALB_NORMALIZE = 9
    ALB_TRANSPOSE = 10
    ALB_RANDOM_CROP = 11
    ALB_RANDOM_GAMMA = 12
    ALB_RANDOM_ROTATE90 = 13
    ALB_ROTATE = 14
    ALB_SHIFT_SCALE_ROTATE = 15
    ALB_CENTER_CROP = 16
    ALB_OPTICAL_DISTORITION = 17
    ALB_GRID_DISTORTION = 18 
    ALB_ELASTIC_DISTORTION = 19 
    ALB_RANDOM_GRID_SHUFFLE = 20
    ALB_HUE_SATURATION_VALUE = 21
    ALB_PAD_IF_NEEDED = 22
    ALB_RGB_SHIFT = 23
    ALB_RANDOM_BRIGHTNESS = 24
    ALB_RANDOM_CONTRAST = 25
    ALB_MOTION_BLUR = 26
    ALB_MEDIAN_BLUR = 27
    ALB_GAUSSIAN_BLUR = 28
    ALB_GAUSSIAN_NOISE = 29
    ALB_GLASS_BLUR = 30
    ALB_CLAHE = 31
    ALB_CHANNEL_SHUFFLE = 32
    ALB_INVERT_IMG = 33
    ALB_TO_GRAY = 34
    ALB_TO_SEPIA = 35
    ALB_JPEG_COMPRESSION = 37
    ALB_IMAGE_COMPRESSION = 38
    ALB_CUTOUT = 39 
    ALB_COARSE_DROPOUT = 40


class Augmenter: 
    def __init__(self, showResult = False, torchMyArray = False):
        """
        args:
            showResult - True for showing image before returning final image
            torchMyArray - True if convert numpy array to tensor for final image
        """
        self.showResult = showResult
        self.torchMyArray = torchMyArray

    def image_agcwd(self,img, a=0.25, truncated_cdf=False):
        h,w = img.shape[:2]
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        prob_normalized = hist / hist.sum()

        unique_intensity = np.unique(img)
        intensity_max = unique_intensity.max()
        intensity_min = unique_intensity.min()
        prob_min = prob_normalized.min()
        prob_max = prob_normalized.max()
        
        pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
        pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
        pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
        prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
        
        if truncated_cdf: 
            inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
        else:
            inverse_cdf = 1 - cdf_prob_normalized_wd
        
        img_new = img.copy()
        for i in unique_intensity:
            img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
       
        return img_new
    def process_bright(self,img):
        img_negative = 255 - img
        agcwd = self.image_agcwd(img_negative, a=0.25, truncated_cdf=False)
        reversed = 255 - agcwd
        return reversed
    def process_dimmed(self,img):
        agcwd = self.image_agcwd(img, a=0.75, truncated_cdf=True)
        return agcwd
    def alb_function(self,augmentationFunc, p, augmentation, image):
        height, width = image.shape[0], image.shape[1]
        if augmentation == Augmentation.ALB_RANDOM_CROP or\
                augmentation == Augmentation.ALB_CENTER_CROP:
            return Compose([augmentationFunc(height = int(.5 * height), width = int(.5 * width), p = p)])
        return Compose([augmentationFunc(p = p)],p = 1 )

    def getAugmentation(self, augmentation):

        if augmentation == Augmentation.ALB_CUTOUT:
            return albumentations.Cutout
        elif augmentation == Augmentation.ALB_IMAGE_COMPRESSION:
            return None # deprecated
            return albumentations.ImageCompression
        elif augmentation == Augmentation.ALB_BLUR:
            return albumentations.Blur
        elif augmentation == Augmentation.ALB_COARSE_DROPOUT:
            return None # deprecated
            return albumentations.CoarseDropout
        elif augmentation == Augmentation.ALB_JPEG_COMPRESSION:
            return albumentations.JpegCompression
        elif augmentation == Augmentation.ALB_TO_GRAY:
            return albumentations.ToGray
        elif augmentation == Augmentation.ALB_TO_SEPIA:
            return None #  deprecated
            return albumentations.augmentations.transforms.ToSepia
        elif augmentation == Augmentation.ALB_INVERT_IMG:
            return albumentations.InvertImg
        elif augmentation == Augmentation.ALB_CHANNEL_SHUFFLE:
            return albumentations.ChannelShuffle
        elif augmentation == Augmentation.ALB_CLAHE:
            return albumentations.CLAHE
        elif augmentation == Augmentation.ALB_GLASS_BLUR:
            return None #  deprecated
            return albumentations.augmentations.transforms.GlassBlur
        elif augmentation == Augmentation.ALB_GAUSSIAN_NOISE:
            return albumentations.GaussNoise
        elif augmentation == Augmentation.ALB_MOTION_BLUR:
            return albumentations.MotionBlur
        elif augmentation == Augmentation.ALB_MEDIAN_BLUR:
            return albumentations.MedianBlur
        elif augmentation == Augmentation.ALB_RANDOM_BRIGHTNESS:
            return albumentations.RandomBrightness
        elif augmentation == Augmentation.ALB_RANDOM_CONTRAST:
            return albumentations.RandomContrast
        elif augmentation == Augmentation.ALB_VERTICAL_FLIP:
            return albumentations.VerticalFlip
        elif augmentation == Augmentation.ALB_HORIZFLIP:
            return albumentations.HorizontalFlip
        elif augmentation == Augmentation.ALB_NORMALIZE:
            return albumentations.Normalize
        elif augmentation == Augmentation.ALB_FLIP:
            return albumentations.Flip
        elif augmentation == Augmentation.ALB_TRANSPOSE:
            return albumentations.Transpose
        elif augmentation == Augmentation.ALB_RANDOM_CROP:
            return albumentations.RandomCrop
        elif augmentation == Augmentation.ALB_RANDOM_GAMMA:
            return albumentations.RandomGamma
        elif augmentation == Augmentation.ALB_RANDOM_ROTATE90:
            return albumentations.RandomRotate90
        elif augmentation == Augmentation.ALB_ROTATE:
            return albumentations.Rotate
        elif augmentation == Augmentation.ALB_SHIFT_SCALE_ROTATE:
            return albumentations.ShiftScaleRotate
        elif augmentation == Augmentation.ALB_CENTER_CROP:
            return albumentations.CenterCrop
        elif augmentation == Augmentation.ALB_OPTICAL_DISTORITION:
            return albumentations.OpticalDistortion
        elif augmentation == Augmentation.ALB_GRID_DISTORTION:
            return albumentations.GridDistortion
        elif augmentation == Augmentation.ALB_ELASTIC_DISTORTION:
            return albumentations.ElasticTransform
        elif augmentation == Augmentation.ALB_RANDOM_GRID_SHUFFLE:
            return None #  deprecated
            return albumentations.augmentations.transforms.RandomGridShuffle
        elif augmentation == Augmentation.ALB_HUE_SATURATION_VALUE:
            return albumentations.HueSaturationValue
        elif augmentation == Augmentation.ALB_PAD_IF_NEEDED:
            return albumentations.PadIfNeeded
        elif augmentation == Augmentation.ALB_RGB_SHIFT:
            return albumentations.RGBShift

    def alb_augmentation(self,image, augmentation = None, p = 1):
        data = {"image": image}
        augFunc = self.getAugmentation(augmentation)
        if augFunc == None: return image
        augFunc = self.alb_function(augFunc, p, augmentation, image)
        augmentedImage = augFunc(**data)
        return augmentedImage["image"]
    def alb_augmentSTANDALONE(image, augmentationClass):
        data = {"image": image}
        augFunc = self.alb_function(augFunc, p, augmentation, image)
        augmentedImage = augFunc(**data)
        return augmentedImage["image"]

        


            

    def IAGCWD(self,image, p = .5):
        """
        https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py
        args: 
            image - ndarray for image to be contrast enhanced
            p - probability that enhancement will be done
        returns: Contrast enhanced image 
        """
        if random.uniform(0, 1) > p:
            return image
        YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        Y = YCrCb[:,:,0]
        threshold = 0.3
        exp_in = 112
        M,N = image.shape[:2]
        mean_in = np.sum(Y/(M*N))
        t = (mean_in - exp_in) / exp_in
        img_output = None
        if t < -threshold: # Dimmed Image
            result = self.process_dimmed(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        elif t > threshold:
            result = self.process_bright(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        else:
                img_output = image
        return img_output
            
    def randomErase(self,
            img,
            p = .5,
            sl = .02,
            sh = .4,
            r1 = .3,
            mean = [.4914,.4822,.4465]):
        """
        https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
        args:
            p - probability that array will have random erasing
            s1 - min erasing area
            sh - max erasing area
            r1: min aspect ratio
            mean: erasing value
        returns - image with randomly erased rectangles
        """
        if random.uniform(0, 1) > p:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
       
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1+h, y1:y1+w,0] = mean[0]
                    img[ x1:x1+h, y1:y1+w,1] = mean[1]
                    img[x1:x1+h, y1:y1+w,2] = mean[2]
                else:
                    img[x1:x1+h, y1:y1+w,0] = mean[0]
                return img
        return img

    def random_crop(self,image, cropRatio = .5,p = .5):
        """
        https://github.com/chainer/chainercv/blob/master/chainercv/transforms/image/random_crop.py
        args: 
            p - probability that array will have random erasing
            image - numpy image array
            cropRatio - height and width will be randomly reduced from 
            .5 * S to 1, where S is either the width or height
        returns: randomly cropped image array
        """
        img = image.copy()
        if random.uniform(0, 1) > p:
            return img
        HRand = r.uniform(cropRatio,1)
        WRand = r.uniform(cropRatio,1)
        H,W = int(HRand * image.shape[0]), int(WRand * image.shape[1])
        if img.shape[1] >= W:
            y_offset = random.randint(0, img.shape[1] - W)
        else:
            raise ValueError('shape of image needs to be larger than output shape')
        y_slice = slice(y_offset, y_offset + W)

        if img.shape[0] >= H:
            x_offset = random.randint(0, img.shape[0] - H)
        else:
            raise ValueError('shape of image needs to be larger than output shape')
        x_slice = slice(x_offset, x_offset + H)

        img = img[ x_slice, y_slice,:]
        return img
    def augmentBatchMul( self, images, augmentations, probabilities):
        
        """
        Params:
               augmentations - list of augmentations to be performed. The augmentations        are chosen from Augmentation enum
               probabilities - list of float probabilities for each augmentation. 
               augmentations list and probabilities list should be the same length
        Returns: returns augmented images 
        """
        newImages = [self.augmentMultiple(image, augmentations,probabilities) for image in images] 
        return newImages
        
    def augmentMultiple(self,image, augmentations, probabilities, copy = True):
        """
        Params:
               augmentations - list of augmentations to be performed. The augmentations        are chosen from Augmentation enum
               probabilities - list of float probabilities for each augmentation. 
               augmentations list and probabilities list should be the same length
        Returns: returns augmented image 
        """
        output = image.copy()
        for i in range(len(augmentations)):
            output = self.augment(output, augmentations[i], probabilities[i],copy = False)
        return output
    def augment(self,image,
            augmentation,
            p = 1,
            sl = .02,
            sh = .4, 
            r1 = .3,
            mean = [.4914,.4822,.4465],
            cropRatio = .5,
            copy = True,

            ):
        """
        args: 
            image - numpy image array
            augmentation - augmentations with choices in augmentation enum
            args for Random Erasing:
                s1 - min erasing area
                sh - max erasing area
                r1: min aspect ratio
                mean: erasing value
            cropRatio - ratio to crop image by
            copy - true if using copy of image or image itself

        returns: image array with applied augmentaion
        """
        output = image.copy() if copy else image
        if augmentation == Augmentation.AGCWD:
            output =  self.IAGCWD(output, p ) 
        elif augmentation == Augmentation.RANDOM_ERASE: 
            output =  self.randomErase(output, p,sl,sh,r1,mean)
        elif augmentation == Augmentation.FLIP_HORIZONTAL:
            if random.uniform(0, 1) > p:
                return img
            outPIL = Image.fromarray(output)
            outPIL = ImageOps.mirror(outPIL)
            output = np.array(outPIL)
        elif augmentation == Augmentation.RANDOM_CROP:
            output = self.random_crop(output,cropRatio, p)
        else:
            output = self.alb_augmentation(image, augmentation, p)


        if self.showResult: 
            outputPIL = Image.fromarray(output.astype(np.uint8))
	
            outputPIL.show()

        if self.torchMyArray: 
            output = torch.from_array(output)
        return output



