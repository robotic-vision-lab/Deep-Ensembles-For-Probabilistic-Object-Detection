import cv2
from PIL import Image
class DetectionPlotter:
    def addBoxesAndText(self, image, boxes, labels):
        """
        Params: 
            image - numpy image array ( h,w,c)
            boxes - list of boxes where each box is in format (x1,y1,x2,y2)
            labels - labels for each box
        Returns - numpy image with boxes overlayed boxes and labels
        
        Reference: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/

        """
        img = image.copy()

        for i in range(len(boxes)):
            cv2.rectangle(img, tuple(boxes[i][0:2]), tuple(boxes[i][2:]), color = (0,255,0), thickness = 2)
            
            #cv2.putText(img, tuple(labels[i]), tuple(boxes[i][0:2]), cv2.FONT_HERSHEY_SIMPLEX, (0,255,0), thickness = 3)
        return img
    def showImageWithBoxesAndText(self, image, boxes, labels):
        """
        Params: 
            image - numpy image array ( h,w,c)
            boxes - list of boxes where each box is in format (x1,y1,x2,y2)
            labels - labels for each of the boxes
        Output: shows image with boxes and labels
        
        Reference: https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
        """
        image = self.addBoxesAndText(image, boxes, labels)
        self.showImage(image)
    def showImagesWBT(self,images, predictions):
        """
        Params:
            images - list of numpy images in (h,w,c) format
            predictions - list of boxes, labels, scores tuples
        output: shows all images with boxes
        """
        labeledImages = self.addBoxesAndTextToImages(images, predictions)
        for labeledImage in labeledImages:
            self.showImage(labeledImage)
    def addImageWBTMultiplePredictions(self, image,predictions):
        """
        Params:
            image - numpy image
            predictions - list of predictions to be added to images
        Returns - list of similar images with different predictions

        """
        images = [image for i in range(len(predictions))] 
        labeledImages = self.addBoxesAndTextToImages(images, predictions)
        return labeledImages
    def showImagesWBTMulPreds(self,image, predictions):
        """
        Params:
            image - numpy image
            predictions - list of predictions to be added to images
        output - shows  similar images with different predictions
        """
        labeledImages = self.addImageWBTMultiplePredictions(image,predictions)
        for labeledImage in labeledImages:
            self.showImage(labeledImage)

    def showImage(self, image):
        """
        Params:
            image - numpy array to be shown
        Output: shows image using PIL
        """
        img = Image.fromarray(image)
        img.show()
    def addBoxesAndTextToImages(self,images, predictions):
        """
        Params: 
            images = list of numpy images to add boxes and text
            predictions - list of boxes, labels scores tuples
        Returns - list of images with boxes and labels
        """
        labeledImages = []
        for i in range(len(images)):
            boxes, labels,_ = predictions[i]
            labeledImage = self.addBoxesAndText(images[i], boxes, labels)
            labeledImages.append(labeledImage)
        return labeledImages

        
        

