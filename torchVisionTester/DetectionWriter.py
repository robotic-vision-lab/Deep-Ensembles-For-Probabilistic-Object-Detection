import json
import pdb
from torchVisionTester.class_list import *
from torchVisionTester.Debugger import Debugger as De
e,d = exec, De()
e(d.gIS())
class DetectionJSON:
    def __init__(self):
        self.class_list = self.readLines('class_list.txt')
        self.classIndices = self.getClassIndicesFromCoco()
        
    def writeJSON(self, data, fileName):
        """
        Params:
            data - dict to be written to file in json
            fileName - name of json file to be produced
        output: saves file to fileName.json holding data in json

        """
        with open(fileName + '.json', 'w') as f:
            json.dump(data,f)
    def readJSON(self, fileName):
        """
        Params:
            fileName - name of json file to be loaded
        Returns: dict loaded from fileName.json
        """
        with open(fileName + '.json') as f:
            data = json.load(f)
            return data
    def readLines(self, fileName):
        """
        Params: 
            fileName - name of file to be loaded into list
        Returns - list of lines in file
        """
        return [line.rstrip('\n') for line in open(fileName)]
    def readCocoLabels(self):
        """
        Params:
            None
        Returns: list of lines in 'cocoLabels.txt

        """
        return self.readLines('cocoLabels.txt')
    def findIndexClassFromClass(self,classItem, class1List,class2List):
        """
        Params:
            classItem - class of item to be found in integer form.
            class1List - list of classes to be identified
            class2List - list of classes to be identified where this class is a superset of class1List
        Returns: First the string name of classItem is found from class1List. Then the corresponding index of that class in class2List is found by searching throough class2List until that ID is found. The index of classItem in the second list is returned.
        """
        
        classString = class1List[classItem -1]
        for i in range(len(class2List)):
            if classString == class2List[i]:
                break
        if i == len(class2List): return None
        return i + 1
        return [classList2IDs[c_Name] for c_Name in classList1]

    def findIndexCocoFromPOD(self, classItem):
        cocoLabels = self.readCocoLabels()
        return self.findIndexClassFromClass(classItem, cocoLabels, CLASSES)


    def getClassIndices(self, classList1, classList2):
        """
        Params: 
            classList1 - list of classes from first list
            classList2 - list of classes from second list
        Returns: indices of each classList1 class in classList2
        """
        classList2IDs = {c_Name: idx for idx, c_Name in enumerate(classList2,1)}
        classIndices = []
        for c_Name in classList1:
            if c_Name == 'none':
                continue
            elif c_Name == 'television':
                classIndices.append(classList2IDs['tv'])
            else: classIndices.append(classList2IDs[c_Name])
        return classIndices

    def getClassIndicesFromCoco(self):
        """
        Params: 
            None
        Returns: list of indices of class labels in coco labels file
        """
        cocoLabels = self.readCocoLabels()
        class_list = self.readLines('class_list.txt')

        return self.getClassIndices(class_list, cocoLabels)
    def makeDetectionJSON(self,fileName,detections):
        """
        Params: 
            fileName - name of json file
            detections - detections for current image
        output: - saves 
        returns: - dict which can be directly converted to JSON
        """
        myDict = {}
        myDict["classes"] = self.class_list
        
        myDict["detections"] = self.makeDetectionList(detections)
        self.writeJSON(myDict, fileName)
        return myDict
    def getNumberedFileNames(self, numberFiles, padding):
        """
        Params:
            numberFiles - number of names of files
            padding - length of number that string represents
        Returns: ordered numberFiles strings of length padding where each string consists of a number
        """
        return [str(i).zfill(padding) for i in range(numberFiles)]
    def makeDetectionJSONMul(self, fileNames, dataList):
        """
        Params: 
            fileNames - list of names for json files
            dataList - list of list of detections. Each inner list is placed in its own json file.
        output: writes len(fileNames) json files containing data from dataList
        """
        for i in range(len(fileNames)):
            self.makeDetectionJSON(fileNames[i], dataList[i])




    def makeDetectionList(self,detections):
        """
        Params:
            detections - list of detections where each  detection is a list of bounding boxes and their corresponding information
        Returns: list of detections in following format:

         "detections":
            [
                [
                  {
                    "bbox": [x1, y1, x2, y2],
                    "covars": [
                      [[xx1, xy1],[xy1, yy1]],
                      [[xx2, xy2],[xy2, yy2]]
                    ],
                    "label_probs": [<an ordered list of probabilities for each class>]
                  },
                  {
                  }
                ],
                [],
                []
            ]
        

        """
        detectList =  [self.makeDictListForDetection(detection) for detection in detections]
        pdb.set_trace()
        return detectList


    def makeDictListForDetection(self, detection):
        """
        Params:
            detection - list of bboxs in following format:
            (c1,c2,label_probs,bbox)
        Returns: detection in following format:
            "detections":
            [
                [
                  {
                    "bbox": [x1, y1, x2, y2],
                    "covars": [
                      [[xx1, xy1],[xy1, yy1]],
                      [[xx2, xy2],[xy2, yy2]]
                    ],
                    "label_probs": [<an ordered list of probabilities for each class>]
                  },
                  {
                  }
                ],
                [],
                []
            ]
            where the list of dicts is what is returned.
        """

        if len(detection) == 0: return []
        for bboxinfo in detection:
            bboxinfo["label_probs"] = [0] + [bboxinfo["label_probs"][i] for i in self.classIndices]
        return detection                
                



