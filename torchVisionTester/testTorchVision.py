import torch
import torchvision.models as models
from torchvision.ops import nms
from torch import nn
from torchVisionTester.DetectionPlotter import DetectionPlotter
from torchVisionTester.DataLoader import *
from torchVisionTester.Debugger import Debugger
from torch import stack as torchStack
import numpy as np
from torchVisionTester.Uncertainty import CovCalculator
from torchVisionTester.Augmenter import Augmentation as AS
from torchVisionTester.DetectionWriter import *
import traceback
e,d = exec, Debugger()
e(d.gIS())
import argparse
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--dropoutRate', type = float, default = 0.1)
parser.add_argument('--dropoutPasses', type = int, default = 5)
parser.add_argument('--imageLimit', type = int, default = 10000)
args = parser.parse_args()
class runModel:
    def __init__(self, model,  dataLoader, jsonWriter):
        """
        This class assumes model is made as in 
        pytoch.org/docs/stable/torchvision/models.html
        Params:
            model - pytorch model from torchvision
            dataLoader - DataLoader object used for loading and augmenting data
            
        Output: Returns model which can predict bounding boxes and labels for dataset
        """
        self.model = model
        self.dl = dataLoader
        self.jsonWriter = jsonWriter

    def ConfidenceFilter(self,prediction,confidence_threshold = 0.8):
        """
        Params:
            confidence_threshold : The score below which all boxes are not included
        Returns:
            Predictions filtered by confidence scores
        """
        detection = prediction[0]
        scores,boxes,labels,softmax_scores = detection["scores"],detection["boxes"],detection["labels"],detection["softmax_scores"]
        confidence_vector = (scores > confidence_threshold)
        scores,boxes,labels = scores[confidence_vector],boxes[confidence_vector],labels[confidence_vector]
        softmax_scores = [value for index,value in enumerate(detection["softmax_scores"]) if confidence_vector[index]]
        return [{"scores":scores,"boxes":boxes,"labels":labels,"softmax_scores":softmax_scores}]
        

    def NMS(self,predictions,threshold):
        """
        Params:
            predictions - The predictions made by the model
            threshold - The IoU threshold for performing non-maximum suppression
        Returns: a list of integer indexes of detections to keep
        """
        detection = predictions[0]
        keep = nms(detection["boxes"],detection["scores"],threshold)
        nms_boxes = detection["boxes"][keep]
        nms_labels = detection["labels"][keep]
        nms_scores = detection["scores"][keep]
        nms_softmax_scores = [value for index,value in enumerate(detection["softmax_scores"]) if index in keep]
        prediction = [{"boxes":nms_boxes,"scores":nms_scores,"labels":nms_labels,"softmax_scores":nms_softmax_scores}]
        return prediction

    def predict(self,fileName,threshold = 0.5):
        """
        Params: 
            fileName - string path to fileName which is an image to be loaded as a pytorch tensor
            threshold - IoU Threshold for Non-Maximum suppression
        Returns: list of single pytorch tensor in format ready for pytorch predictions
        """
        #self.model.eval() # Switch to inference mode
        batch = self.dl.getArraysFromFileNames([fileName], ClassTypes.PYTORCH_TENSOR, augment = True)
        
        prediction = self.model(batch)
        prediction = self.NMS(prediction,threshold)
        prediction = self.ConfidenceFilter(prediction,0.0)
        moveDetectionCPU(prediction) 
        return prediction
    def predictFilenames(self, fileNames, threshold = 0.5):
        """
        Params:
            fileNames - list of  string paths to fileNames which are images to be loaded as  pytorch tensors
            threshold - IoU Threshold for Non-Maximum suppression
        Returns: list of single pytorch tensor in format ready for pytorch predictions
        """
        batch = self.dl.getArraysFromFileNames(fileNames, ClassTypes.PYTORCH_TENSOR, augment = True)
        predictions = self.model(batch)
        predictions = [self.NMS(prediction,threshold) for prediction in predictions]
        predictions = [self.ConfidenceFilter(prediction,0) for prediction in predictions]
        for pred in prediction: moveDetectionCPU(pred)
        return predictions


       
    def create_json(self, detections):
        return self.jsonWriter.makeDetectionJSON('000000',detections)
        

    def testModel(self, directories):
        """
        Params: 
            directories - list of paths to directories holding images
        Returns: lists of bounding boxes, labels, and scores for all images in tuple
        """
        predictions = [] 
        fileNames = self.dl.getFilePathsFromDirectories(directories,['jpg','png','jpeg'])
        predictions = [self.predict(fileName) for fileName in fileNames]
        return predictions
    def testModelWithFiles(self,fileNames):
        """
        Params:
		fileNames - list of paths to filenames to predict
        Returns: lists of bounding boxes, labels, and scores for all images in tuple
        """
        predictions = [self.predict(fileName) for fileName in fileNames]
        return predictions
    def testDropout(self, fileName, numberPasses):
        """
        Params:
            fileName - path to image to be loaded
            numberPasses - number of times to add dropout
        
        """
        predictions = [self.predict(fileName,threshold=0.5) for i in range(numberPasses)]

        return predictions
    def testMulDropout(self, fileNames, numberPasses):
        """
        Params: 
            fileNames - list of paths to images to be predicted
            numberPasses - number of times to pass image through model with dropout
        Returns: list of lists where inner lists are numberPasses predictions for a single image. Outer list represents predictions for multiple images.

        """
        return [self.testDropout(fileName, numberPasses) for fileName in fileNames]
    def testDropoutOnDirectories(self, directories, numberPasses, imageLimit = 2):
        """
        Params:
            directories - list of paths to directories containing images
            numberPasses - number of times to pass images through network with dropout.
        Returns - lists of lists of lists  of predictions where third list corresponds to dropout ensembles, second list corresponds to different images, and first list corresponds to different directories
        """
        predictionListDir = []
        for directory in directories:
            fileNames = self.dl.getFilePathsFromDirectory(directory, ['png'])
            fileNames = fileNames[:imageLimit]
            predictionListDir.append(self.testMulDropout(fileNames, numberPasses))
        return predictionListDir
    def printPredictions(self, predictions):
        """
        Params:
            predictions - the predictions returned by the model
        """
        for idx,prediction in enumerate(predictions):
            detection  = prediction[0]
            print("Boxes : {} \n Labels : {} \n Scores : {}".format(detection["boxes"],detection["labels"],detection["scores"]))

def printModel( model):
    print(dir(model))
def moveDetectionCPU(prediction):
    prediction[0]['boxes'] =prediction[0]['boxes'].to('cpu').detach()
    prediction[0]['labels'] =prediction[0]['labels'].to('cpu').detach()
    prediction[0]['scores']= prediction[0]['scores'].to('cpu').detach()
    sc = prediction[0]['softmax_scores']
    prediction[0]['softmax_scores']= [sc[i].cpu().detach() for i in range(len(sc))]
    
def convertPredictionsList( predictions):
    newpreds = []
    for prediction in predictions:
        myDict = {}
        pred = prediction[0]
        myDict['boxes'] = pred['boxes'].numpy().tolist()
        myDict['labels'] = pred['labels'].numpy().tolist()
        myDict['softmax_scores'] = [score.tolist() for score in pred['softmax_scores']]

        newpreds.append(myDict)
    return newpreds

def makePredictionsData(boxes,scores):
    """
    Params: 
        boxes - boxes from output of model in tensor form
        scores - softmax scores for each box
    Returns: list of bounding boxes and their corresponding infos
    """
    num_detected_boxes = boxes.shape[-1]

    # obtain covariances for corners of bounding boxes
    detectionData = []
    for box_no in range(num_detected_boxes):
        lists = boxes[:,:,box_no]
        lists_softmax = scores[:,:,box_no]
        tuples = [tuple(box_coords) for box_coords in lists.tolist()]
        corner1covariance, corner2covariance = calc.getBoxCovariances(tuples)
        label_probs = np.mean(lists_softmax,axis = 0)
        bbox = np.mean(lists,axis = 0)
        detectionData.append((corner1covariance.tolist(), corner2covariance.tolist(),label_probs.tolist(), bbox.tolist()))
    return detectionData
def isEmptyPrediction(predictions):
    for prediction in predictions:
        box = prediction[0]['boxes']
        e(g('box'))
        e(g('len(box)'))
        if len(prediction[0]['boxes']) > 0: 
            return False
    return True
        
def makeData(predictions, numDropout, num_classes):
    """
    Params: 
        predictions - list of predictions from model, where all predictions are from passing a single image multiple times with dropout
        numDropout - number of times dropout is applied for each prediction
        num_classes - number of classes to detect
    Returns: list of bounding boxes and their info for this image
    """
    if isEmptyPrediction(predictions): return None
    boxes, scores = calc.getNumpyArrayfromPredictions(predictions, numDropout, num_classes)
    return makePredictionsData(boxes, scores)
def mulMakeData(listOfDropoutPreds, numDropout, num_classes):
    """
    Params:
        listOfDropoutPreds - list of lists of predictions where all predictions are from passing a single image multiple times with dropout. The outer list represents different images.
        numDropout - number of times dropout is applied for each prediction
        num_classes - number of classes to detect
    Returns: list of lists of bounding boxes and their info for multiple images. 
    """
    
    return [makeData(predictionList, numDropout, num_classes) for predictionList in listOfDropoutPreds]
def mulmakeDataDir(listOfPredsDirs, numDropout, num_classes):
    """
    
    Params:
        listOfDropoutPreds -  list of list of lists of predictions where all predictions are from passing a single image multiple times with dropout. The 2nd outer list represents different images. The most outer list corresponds to different directories
        numDropout - number of times dropout is applied for each prediction
        num_classes - number of classes to detect
    Returns:  list of list of lists of bounding boxes and their info for multiple images. Outer most list corresponds to different directories
    """
    return [mulMakeData(dirPredList, numDropout, num_classes) for dirPredList in listOfPredsDirs]
def getAllBoxes( predictions):
    """
    Params: 
        predictions - list of predictions
    Returns - obtains list of all bounding box, softmax_scores tuples.  
    """
    boxes = []
    scores = []
    for prediction in predictions: 
        boxes += prediction['boxes']
        scores += prediction['softmax_scores']
    boxScores = [(boxes[i],scores[i]) for i in range(len(boxes))]
    return boxScores 

def IOU(box1, box2):
    """
    Params
        box1 - [x1,y1,x2,y2] which are the coordinates of the top left and bottom right  corners of a box
        box2 - [x1,y1,x2,y2] which are the coordinates of the top left and bottom right  corners of a box
    Returns - Intersection over union of the two bounding boxes
    """
    x_left = max( box1[0], box2[0])
    y_top = max( box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: 
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1area = (box1[2] - box1[0]) * ( box1[3] - box2[1])
    box2area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1area + box2area - intersection_area)
    return iou

def getObservations(predictions, T = .80, conf = .4):
    """
    Params: 
        predictions - list of predictions in list form ready for being json
        T - Threshold to group boxes by IOU
    Returns - list of bounding boxes with covariance and softmax scores. They are grouped according to IOU scores. This reduces the number of boxes by on average a factor of len(predictions)
    """
    boxScores = getAllBoxes(predictions) 
    observations = []
    while len(boxScores) > 0: 
        if boxScores[-1] != None:
            box, score = boxScores.pop()
        else:
            boxScores.pop()
            continue
        if max(score) < conf: continue
        observation = []
        observation.append((box,score))
        removeIndices =[]
        for i in range(len(boxScores)):
            if boxScores[i] is not None:
                otherbox, otherscore = boxScores[i] 
                iou = IOU(box,otherbox)
                if iou > T:
                    observation.append((otherbox,otherscore))
                    boxScores[i] = None
        observations.append(observation)
    return observations

def ProcessObservations(observations,num_classes):
    averageObservations = []
    for observation in observations:
        tuple_list = []

        boxes_array = np.empty((0,4))
        scores_array = np.empty((0,num_classes))
        for forwardpass in range(len(observation)):
            boxes,scores = observation[forwardpass]
            boxes_array = np.vstack((boxes_array,np.array(boxes)))
            scores_array = np.vstack((scores_array,np.array(scores).reshape(1,-1)))
            tuple_list.append(tuple(boxes))
        mean_coords = np.mean(boxes_array,axis = 0)
        mean_softmax = np.mean(scores_array,axis = 0)   # Noticed that we are getting NaN values here, possible numerical instability
        if len(observation) == 1: covs = calc.getBoxFixedCov(3) 
        else: covs = calc.getBoxCovariances(tuple_list)
        averageObservation = {}
        averageObservation["bbox"] = mean_coords.tolist()
        averageObservation["covars"] = covs
        averageObservation["label_probs"] = mean_softmax.tolist()
        averageObservations.append(averageObservation)
    return averageObservations
def testDict(filename, numDropout):
    """
    Params:
        filename - This is the path to the image. We will predict the bounding boxes bases on this image
        numDropout - number of times each image a forward pass is obtained with dropout
    Returns - prediction in the form of a list of dicts. Each dict has bounding box corners, cov matrix for each corner, and softmax scores. These three pieces of information are identifies by  dict keys.
    """
    predictions = run.testDropout( filename, numDropout)
    predictions = convertPredictionsList(predictions)
    observations = getObservations(predictions)
    averageObservations = ProcessObservations(observations, 90)
    return averageObservations
def testMulDict(filenames, numDropout,jsonname = None, writeJSON = False, jsonWriter = None):
    """
    Params:
        filenames - list of paths to image
        numDropout - number of times each image a forward pass is obtained with dropout
        jsonname - name of json file which is only applicable if writeJSON is true
        writeJSON - boolean, if true then detections are written to jsonname.json
    Returns - list of predictions where each prediction is a list of dicts. Each dict has bounding box corners, cov matrix for each corner, and softmax scores. These three pieces of information are identified by dict keys.
    """
    assert jsonWriter != None
    detections= [testDict(filename,numDropout) for filename in filenames]
    if writeJSON: 
        jsonWriter.makeDetectionJSON(jsonname, detections)
    return detections
def testDirMulDict(directories, numDropout, jsonnames, imageLimit, dl, jsonWriter):
    """
    Params: 
        directories - list of directories containing images
        numDropout - number of times each image a forward pass is obta
        ined with dropout
        jsonnames - names of json files to be written, one for each directory
    Output - writes all predictions in json format  using jsonnames where there is one jsonname for each directory in directories
    """
    idx = 0
    for directory in directories:
        e(g('directory'))
        fileNames = dl.getFilePathsFromDirectory(directory, ['png'])
        fileNames = fileNames[:imageLimit]
        testMulDict(fileNames, numDropout, jsonnames[idx], writeJSON = True, jsonWriter = jsonWriter)
        idx +=1






def testDirJSON(directories,
                numPasses,
                numClasses,
                jsonNames,
                imageLimit,
                jsonWriter,
                run,):
    idx = 0
    batch_size = 16
    for directory in directories: 
        e(g('directory'))
        fileNames = dl.getFilePathsFromDirectory(directory, ['png'])
        fileNames = fileNames[:imageLimit]
        detectData = []
        batchNum = 0
        
        while len(fileNames) > 0:
            e(g('batchNum'))
            batch = fileNames[:batch_size]
            fileNames = fileNames[batch_size:]
            batchPredList = run.testMulDropout(batch, numPasses)
            #try:
            batchData =  mulMakeData(batchPredList, numPasses, numClasses)
            """
            except Exception :
                for i in range(len(batchPredList)):
                    try:
                        makeData(batchPredList[i], numPasses, numClasses)
                    except: pdb.set_trace()
            """
            detectData += batchData 
            batchNum +=1


        jsonWriter.makeDetectionJSON(jsonNames[idx], detectData)
        idx += 1




def test(model):
    print("done")
    model.eval()


    # create jsonWriter object
    jsonWriter = DetectionJSON()
    calc = CovCalculator()
    # Get directories for testing
    jsonNames = jsonWriter.getNumberedFileNames(17, 6)
    dp = DetectionPlotter()
    dd = DatasetDirectories()
    directories = dd.getDirectories(Datasets.POD_TEST)
    dl = DataLoader()
    datasetFiles = dl.getFilePathsFromDirectories(directories,['png'])
    dataset = datasetFiles[:2]
    # shrink dataset for testing

    # number of times to pass through dropout
    numDropoutEnsemble = args.dropoutPasses
    imageLimit = args.imageLimit
    numClasses = 91
    # declare augmentations and probabilities of applying them
    augmentations = [AS.AGCWD,AS.ALB_FLIP, AS.ALB_RGB_SHIFT]
    probabilities = [1,0,0]
    dl = DataLoader( augmentations, probabilities)

    run = runModel(model, dl, jsonWriter)
    #Remove extra dimensions from predictions
    #and obtain covariances for corners of bounding boxes 
    
    #dirDetectData = testDirJSON(directories, numDropoutEnsemble, numClasses,jsonNames, imageLimit)
    testDirMulDict([directories[0]], numDropoutEnsemble, jsonNames, imageLimit, jsonWriter)
    #Plot results
    #convPred = convertPredictionsNumpy(predictionList[0])
    img = np.asarray(Image.open(dataset[0]))

    img = dl.getArraysFromFileNames([dataset[0]], ClassTypes.NUMPY, stack = False, augment = True) 
    #dp.showImagesWBTMulPreds(img[0], convPred)

