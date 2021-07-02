import numpy as np
import torch
from numpy import stack, cov, zeros
from torchVisionTester.Debugger import Debugger as De
import pdb
e,d = exec, De()
e(d.gIS())
def is_positive_definite(mat):
    """
    Check if a matrix is positive semi-definite, that is, all it's eigenvalues are positive.
    All covariance matrices must be positive semi-definite.
    Only works on symmetric matrices (due to eigh), so check that first
    :param mat:
    :return:
    """
    if np is not None:
        mat = np.asarray(mat)
        eigvals, _ = np.linalg.eigh(mat)
        return np.all(eigvals >= -1e-14)
    # Numpy is unavailable, assume the matrix is valid
    return True

class CovCalculator:
    
    def getCovariance(self, varLists):
        """
        Params:
            varLists - list of lists where each of the inner lists has the same 
            length. Each of the outer lists represents a different random 
            variable. The inner list holds samples of a particular random 
            variable.
        Returns: covariance matrix of len(varLists) random variables
        """
        X = stack(varLists, axis = 0)
        covar = cov(X)
        return cov(X)
    def getFixedCovariance(self, numberVars, value):
        """
        Params: 
            numberVars - number of random variables for covariance matrixes
            value - value along diagonal of covariance matrix
        Returns - diagonal numpy matrix where the diagonnal has values each with 
        value while all other elements are zero
        """
        covArray = zeros((numberVars,numberVars))
        for i in range(covArray.shape[0]):
            covArray[i][i] = value
        return covArray.tolist()
    def getNumFixedCovariance(self, numberVars, value, num):
        """
        Params: 
            numberVars - number of random variables for covariance matrixes
            value - value along diagonal of covariance matrix
        Returns - list of num diagonal numpy matrices where the diagonnal has values assigned value
        """
        covArrays = [self.getFixedCovariance(numberVars, value) for i in range(num)]
        return covArrays

    def getBoxFixedCov(self,  value):
        """
        Params:
            value - value along diagonal of covariance matrix
        Returns - 2 diagonal numpy matrices where the diagonal has values asigned value
        """
        covArrays = [self.getFixedCovariance(2, value) for i in range(2)]
        return covArrays
    def getListsFromTupleList(self,TupleList):
        """
        Params:
            TupleList - list of tuples of arbitrary length. Each tuple is of 
            length tupleLength
        Returns - tupleLength lists where each list has length len(TupleList). 
        This function splits up TupleList by the number of elements in one it
        its tuples
        """
        Lists = ()
        for i in range(len(TupleList[0])):
            myList = [element[i] for element in TupleList]
            Lists += (myList,)

        return Lists
    def getBoxCovariances(self,boxes):
        """
        Params:
            boxes - list of tuples where each tuple is in the following format:
                (x1,y1,x2,y2)
                The first two elements of the tuple are for the top left corner
                of a box while the second two elements are for the bottom right 
                corner
        Returns - Covariance(top left corner), Covariance(bottom right corner) 
        in the form of a tuple
        """
        x1,y1,x2,y2 = self.getListsFromTupleList(boxes)[:4]

        corner1Covariance = self.getCovariance([x1,y1])
        corner2Covariance = self.getCovariance([x2,y2])
        return [corner1Covariance.tolist(), corner2Covariance.tolist()]

    def getNumpyArrayfromPredictions(self,predictions,numDropoutEnsemble,numClasses):

        """
        Params:
            predictions - The predictions list returned from the model
                            Organization:
                                Dimension 1 - numDropoutEnsemble
                                Dimension 2 - 1 (List to contain dictionary)
                                Dimension 3 - {"labels": numDetections labels
                                                "scores": numDetections scores
                                                "boxes": numDetections boxes , each box is a tensor of (1,4)
                                                "softmax_scores": numDetections boxes, each detection contains a tensor of (1,10) softmax labels
                                                }

        Returns:
            Squeezed numpy array to remove the unnecessary dimensions

        """

        predictions = np.array(predictions).squeeze(1)   # Remove the unnecessary first dimension
        e(g('predictions'))

        numDetectionsA = np.array([])
        for i in range(numDropoutEnsemble):
            e(g('numDetectionsA'))
            num = predictions[i]["boxes"].shape[0]
            e(g('num'))
            numDetectionsA = np.append(numDetectionsA,int(predictions[i]["boxes"].shape[0]))

        counts = np.bincount(numDetectionsA.astype(int))
        numDetections = np.argmax(counts)


        detected_boxes = np.array([])
        
        detected_softmax_scores = np.array([])

        for i in range(numDropoutEnsemble):
            extracted_boxes = predictions[i]["boxes"].to('cpu').detach().numpy()
            if extracted_boxes.shape[0] == numDetections and extracted_boxes.shape[0] > 0:
                #pdb.set_trace()
                e(g('extracted_boxes.shape'))
                e(g('numDetections'))
                
                e(g('counts'))
                if numDetections == 0: pdb.set_trace() 
                detected_boxes = np.append(detected_boxes,extracted_boxes)   # Shape (numDetections,4), type: np.array([])
                detected_softmax_scores = np.append(detected_softmax_scores,torch.stack(predictions[i]["softmax_scores"],dim = 0).to('cpu').detach().numpy()) # Shape (numDetections,numClasses - 1), type: np.array([])
            else:
                print("Uneven number of detections found, skipping this prediction")
                numDropoutEnsemble -= 1

        if numDropoutEnsemble == 0: return []
        #e(g('detected_boxes'))
        detected_boxes = np.reshape(detected_boxes,(numDropoutEnsemble,4,-1))
        detected_softmax_scores = np.reshape(detected_softmax_scores,(numDropoutEnsemble,numClasses-1,-1))

        return detected_boxes,detected_softmax_scores
