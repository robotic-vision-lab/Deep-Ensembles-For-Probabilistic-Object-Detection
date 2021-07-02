from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
from PODStarterKit.submission_builder import *
import pdb
import argparse
from torchVisionTester.Uncertainty import CovCalculator
from mmdet.ops.nms.nms_wrapper import nms_match, soft_nms
import torch
from torch import nn
from PIL import Image
from tools.Augmenter import *
import numpy as np
import random
import time
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--probability', type=float, default=1)
parser.add_argument('--xvar', type=int, default=20)
parser.add_argument('--yvar', type=int, default=20)
parser.add_argument('--covPercent', type=float, default=.10)
parser.add_argument('--inflateConfidence', type=bool, default=False)
parser.add_argument('--removeLowScores', type=bool, default=True)
parser.add_argument('--reduceBoxSize', type=bool, default=True)
parser.add_argument('--boxRatio', type=float, default=.05)
parser.add_argument('--resultNumber', type=int, default=0)
parser.add_argument('--useCovPercent', type=bool, default=True)
parser.add_argument('--numFrames', type=int, default=5200)
parser.add_argument('--isSubmitting', type=bool, default=False)  # True if making actual submission. False otherwise.
parser.add_argument('--isEvaluating', type=bool, default=True)
parser.add_argument('--usingEnsembles', type=bool, default=True)
parser.add_argument('--iouThreshold', type=float, default=.8)
parser.add_argument('--machineNumber', type=int, default=1)
parser.add_argument('--machineID', type=int, default=1)
parser.add_argument('--usingAGC', type=bool, default=True)
parser.add_argument('--ensVars', type=bool, default=False)
parser.add_argument('--saveBoxes', type=bool, default=False)
parser.add_argument('--isReusing', type=bool, default=True)


def test(threshold=0.8,
         probability=1,
         xvar=20,
         yvar=20,
         covPercent=0.1,
         inflateConfidence=False,
         removeLowScores=True,
         reduceBoxSize=True,
         boxRatio=0.05,
         resultNumber=0,
         useCovPercent=True,
         numFrames=5200,
         isSubmitting=False,
         isEvaluating=True,
         usingEnsembles=False,
         iouThreshold=.8,
         machineNumber=1,
         machineID=1,
         usingAGC=True,
         ensVars=False,
         saveBoxes=True,
         isReusing=False,
         sameLabels=False,
         usingDropout=False,
         numDropoutPasses=3,
         dropoutRate=0.3,
         cascadeNMS=False,
         boxList=None,
         labelList=None,
         saveExamples=False,
         usingGrid=False):
    args = parser.parse_args()
    threshold = threshold
    probability = probability
    xvar = xvar
    yvar = yvar

    config_file = os.path.join("configs", "cascade_rcnn", "customCascR50.py")
    config_file_retina = os.path.join("configs", "retinanet", "retinanet_x101_64x4d_fpn_1x_coco.py")
    config_file_htc = os.path.join("configs", "htc", "htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py")
    config_file_grid = os.path.join("configs", "grid_rcnn", "grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py")

    checkpoint_file = os.path.join("work_dirs", "customCascR50", "epoch_12.pth")
    checkpoint_htc = os.path.join("checkpoints",
                                  "htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth")
    checkpoint_retina = os.path.join("checkpoints", "retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth")
    checkpoint_grid = os.path.join("checkpoints", "grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth")

    if not isReusing:
        # model = init_detector(config_file_htc, checkpoint_htc, device='cuda:0')
        if usingGrid:
            model = init_detector(config_file_grid, checkpoint_grid, device='cuda:0')
            if usingDropout:
                # model.roi_head.bbox_head.shared_fcs[1] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head.shared_fcs[1])
                model.roi_head.bbox_head.shared_fcs[1] = nn.Sequential(
                    model.roi_head.bbox_head.shared_fcs[1], nn.Dropout(dropoutRate))
                # print(model.roi_head.bbox_head)
                # exit(0)
        else:
            model = init_detector(config_file_htc, checkpoint_htc, device='cuda:0')
            if usingDropout:
                # model.roi_head.bbox_head[0].fc_cls = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[0].fc_cls)  # Special case: error!
                # model.roi_head.bbox_head[0].shared_fcs[0] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[0].shared_fcs[0])  # variant 0_0
                # model.roi_head.bbox_head[0].shared_fcs[0] = nn.Sequential(
                #     model.roi_head.bbox_head[0].shared_fcs[0], nn.Dropout(dropoutRate))  # variant 1_0
                # model.roi_head.bbox_head[0].shared_fcs[1] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[0].shared_fcs[1])  # variant 2_0
                # model.roi_head.bbox_head[0].shared_fcs[1] = nn.Sequential(
                #     model.roi_head.bbox_head[0].shared_fcs[1], nn.Dropout(dropoutRate))  # variant 3_0
                #
                # model.roi_head.bbox_head[1].shared_fcs[0] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[1].shared_fcs[0])  # variant 0_1
                # model.roi_head.bbox_head[1].shared_fcs[0] = nn.Sequential(
                #     model.roi_head.bbox_head[1].shared_fcs[0], nn.Dropout(dropoutRate))  # variant 1_1
                # model.roi_head.bbox_head[1].shared_fcs[1] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[1].shared_fcs[1])  # variant 2_1
                # model.roi_head.bbox_head[1].shared_fcs[1] = nn.Sequential(
                #     model.roi_head.bbox_head[1].shared_fcs[1], nn.Dropout(dropoutRate))  # variant 3_1, original one
                # model.roi_head.bbox_head[1].shared_fcs[0] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[1].shared_fcs[0], nn.Dropout(dropoutRate))  # 0_1+1_1 v2
                # model.roi_head.bbox_head[1].shared_fcs[1] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[1].shared_fcs[1], nn.Dropout(dropoutRate))  # 2_1+3_1 v2
                #
                # model.roi_head.bbox_head[2].shared_fcs[0] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model.roi_head.bbox_head[2].shared_fcs[0])  # variant 0_2
                # model.roi_head.bbox_head[2].shared_fcs[0] = nn.Sequential(
                #     model.roi_head.bbox_head[2].shared_fcs[0], nn.Dropout(dropoutRate))  # variant 1_2
                model.roi_head.bbox_head[2].shared_fcs[1] = nn.Sequential(
                    nn.Dropout(dropoutRate), model.roi_head.bbox_head[2].shared_fcs[1])  # variant 2_2
                model.roi_head.bbox_head[2].shared_fcs[1] = nn.Sequential(
                    model.roi_head.bbox_head[2].shared_fcs[1], nn.Dropout(dropoutRate))  # variant 3_2

                # print(model.roi_head.bbox_head)
                # exit(0)

        if usingEnsembles:
            model2 = init_detector(config_file_grid, checkpoint_grid, device='cuda:0')
            if usingDropout:
                model2.roi_head.bbox_head.shared_fcs[0] = nn.Sequential(
                    nn.Dropout(dropoutRate), model2.roi_head.bbox_head.shared_fcs[0])  # v0
                # model2.roi_head.bbox_head.shared_fcs[0] = nn.Sequential(
                #     model2.roi_head.bbox_head.shared_fcs[0], nn.Dropout(dropoutRate))  # v1
                # model2.roi_head.bbox_head.shared_fcs[1] = nn.Sequential(
                #     nn.Dropout(dropoutRate), model2.roi_head.bbox_head.shared_fcs[1])  # v2
                # model2.roi_head.bbox_head.shared_fcs[1] = nn.Sequential(
                #     model2.roi_head.bbox_head.shared_fcs[1], nn.Dropout(dropoutRate))  # V3 original one
                # print(model2.roi_head.bbox_head)
                # exit(0)

    boxFileName, labelFileName = getFileName(usingAGC, usingDropout, usingEnsembles, usingGrid)

    classes = ['person', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'orange',
               'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'none']
    classlist = {0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane',
                 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant',
                 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog',
                 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe',
                 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee',
                 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat',
                 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle',
                 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana',
                 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
                 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed',
                 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote',
                 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink',
                 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear',
                 79: u'hair drier', 80: u'toothbrush'}
    keys = {classlist[key]: key for key in classlist.keys()}
    classids = [keys[cls] - 1 for cls in classes[:-1]] + [81]  # [0,39,40,41,42,43,44,45,46,47,49,55,56,57,58,59,60,
    #  61,62,63,64,66,67,68,69,70,71,72,73,74,81]

    if isSubmitting:
        writer = SubmissionWriter('submission', classes)
    else:
        writer = SubmissionWriter('rvchallenge-evaluation-master/det', classes)
    cov = CovCalculator()

    from numpy import linalg as la
    def find_nearest_positive(A):
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = (A + A.T) / 2
        _, s, V = la.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if is_positive_definite(A3):
            return A3
        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not is_positive_definite(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1
        return A3

    def normalize_covariance(cov):
        #dim = np.array(cov).shape[1]
        dim = cov.shape[1]
        I3 = 3.0 * np.eye(dim, dim)
        normA = (1.0 / 3.0) * np.multiply(cov, I3).sum()
        if normA == 0:
            A = np.zeros((2, 2))
        else:
            A = np.divide(cov, normA)
        iterN = 3
        Y = np.zeros((iterN, dim, dim))
        Z = np.array(torch.eye(dim, dim).view(1, dim, dim).repeat(iterN, 1, 1))
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            Y = np.matmul(A, ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y = np.matmul(A, ZY)
            Z = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - np.matmul(Z, Y))
                Y = np.matmul(Y, ZY)
                Z = np.matmul(ZY, Y)
            ZY = 0.5 * np.matmul(Y, I3 - np.matmul(Z, Y))
        if normA <= 0:
            return ZY
        # if normA == 0:
        #     return ZY
        # if normA < 0:
        #     normA *= -1
        else:
            return ZY * np.sqrt(normA)

    def merge_detections(boxes, labels, T, variances=False, same_labels=True, max_box=True):
        """
        :param boxes: list of all boxes from all of the models, shape: (N, 5), [[295.46, 221.31, 313.18, 243.96, 0.0107]
                      , ... , [x1, y1, x2, y2, s]]
        :param T: IoU thresh for NMS. (float)
        :param same_labels: ?
        :param maxBox: a boolean variable indicating using merged box or the box with the highest score
        :return: list of detections where boxes and scores are averaged when boxes of the two separate predictions have high IOU scores.
        """
        # print(boxes)
        matches = nms_match(np.array(boxes), T)  # The outer list corresponds different matched group, the inner Tensor
        # corresponds the indices for a group in score order.[array([87, 50, 18, 78..]), array([71, 75, 16, 12, 47, 43])
        # , array([59,  4, 32, 21, 66, 84]), array([11, 79, 42, 70, 53, 23, 41]), array([61,  5, 33]),
        # array([ 2, 62, 34, 35, 9]), array([ 3, 30, 64, 36, 65,  8]), array([31, 63,  7]), array([52, 19, 80, 45, 68,
        # 74, 14, 40]), array([ 6, 38, 67]), array([57, 27]), array([37]), array([39])]

        # soft_nms
        # if len(np.array(boxes)) == 0:
        #     matches = nms_match(np.array(boxes), T)
        # else:
        #     new_dets, inds = soft_nms(np.array(boxes), T, sigma=0.5)  # use Soft-NMS method
        #     matches = [np.array([ind]) for ind in inds]

        box_list = []
        label_list = []
        var_list = []
        upper_cov_list = []
        lower_cov_list = []
        for match in matches:
            # match[0]:the index of the first box in each matched group, new_boxes are the corresponding box sublist from
            # boxes with indices 'match' in score order.
            # new_boxes: [[x1,y1,x2,y2,s1], [x1,y1,x2,y2,s2], [x1,y1,x2,y2,s3],...]
            new_boxes = [boxes[match[0]]]
            label = labels[match[0]]
            for id in match[1:]:  # the index of remaining boxes in that matched group
                if same_labels:
                    if label == labels[id]:
                        new_boxes.append(boxes[id])
                else:
                    new_boxes.append(boxes[id])
            new_score = np.max([new_box[-1] for new_box in new_boxes])  # equals to new_boxes[0][-1]

            # calculate the covariance
            M = len(new_boxes)
            x_upleft = np.array([new_boxes[i][0:2] for i in range(0, M)])  # (M, 2): [[x1,y1],[x1,y1],...]
            x_bottomright = np.array([new_boxes[i][2:4] for i in range(0, M)])  # (M, 2): [[x2,y2],[x2,y2],...]
            I_hat = (-1. / M / M) * np.ones((M, M)) + (1. / M) * np.eye(M)
            upper_left_cov = np.matmul(np.matmul(x_upleft.transpose(), I_hat), x_upleft)
            lower_right_cov = np.matmul(np.matmul(x_bottomright.transpose(), I_hat), x_bottomright)

            # if upper_left_cov.any() == 0:
            #     norm_upper_left_cov = upper_left_cov
            # else:
            #     norm_upper_left_cov = normalize_covariance(upper_left_cov)
            # if lower_right_cov.any() == 0:
            #     norm_lower_right_cov = lower_right_cov
            # else:
            #     norm_lower_right_cov = normalize_covariance(lower_right_cov)

            # normalization
            norm_upper_left_cov = normalize_covariance(upper_left_cov)
            norm_lower_right_cov = normalize_covariance(lower_right_cov)
            norm_upper_left_cov = np.around(norm_upper_left_cov, 4)  # [[3.03406, 2.36384], [2.36384, 5.25683]]
            norm_lower_right_cov = np.round(norm_lower_right_cov, 4)  # [[1.24515, 49.94519], [49.94519, 2956.4942]]
            #pdb.set_trace()

            upper_cov_list.append(norm_upper_left_cov)
            lower_cov_list.append(norm_lower_right_cov)

            # # pdq score 22.678 using below code block, used when ensVars=True   01/07添加
            # # tested on 2021/01/07 with dropout_3_1, got 20.4623
            # # coordVars = [np.var([new_box[i] for new_box in new_boxes]) for i in range(4)]
            # coordVars = []
            # sim = 0.01
            # for i in range(4):
            #     maxScore = new_boxes[0][-1]
            #     coordinate = []
            #     for new_box in new_boxes:
            #         if abs(new_box[-1] - maxScore) < sim:
            #             coordinate.append(new_box[i])
            #     coordVars.append(np.var(coordinate))
            # maxScore = new_boxes[0][-1]
            # filteredBoxes = []
            # for new_box in new_boxes:
            #     if abs(new_box[-1] - maxScore) < sim:
            #         filteredBoxes.append(new_box)
            # new_boxes = filteredBoxes
            # # end code block

            # 如果不用上面那段代码，需要用下面这个vars --2020/01/07
            vars = [np.var([new_box[i] for new_box in new_boxes]) for i in range(4)]
            # vars: [3.034058995289658, 5.256832560331902, 1.2451500223037237, 2956.4941991608002]...
            boxxs = [new_box[1] for new_box in new_boxes]
            nvars = np.var([new_box[1] for new_box in new_boxes])
            var_list.append(vars)  # var_list has the same size with matches' outer list, there is one problem that when
            # there is only one box in the group in matches, the variance list will be all 0: [0.0, 0.0, 0.0, 0.0].

            # var_list.append(coordVars)  # used only with above code block
            merged_box = np.mean(new_boxes, axis=0)
            if max_box:  # no merge, take the first one in each box list
                merged_box = new_boxes[0]
                merged_box[-1] = new_score
            box_list.append(
                merged_box)  # a list of boxes, where each box is the first box or merged box in matched group
            label_list.append(label)  # [58, 56, 0, 45, 0, 0, 0, 0, 58, 0, 74, 0, 0]
        return box_list, label_list, var_list, upper_cov_list, lower_cov_list

    def getBoxesLabels(detections):
        """
        :param detections:
        :return:
        """
        boxes = []
        labels = []
        for i in range(len(detections)):  # num classes
            boxList = detections[i].tolist()
            for box in boxList:
                if i in classids:
                    boxes.append(box)
                    labels.append(i)
        return boxes, labels

    def clamp(num, low=5, high=100):
        if num < low:
            return low
        elif num > high:
            return high
        else:
            return num

    if isSubmitting:
        testDirectory = os.path.join('E:', '\\test_data\\')
    else:
        testDirectory = os.path.join('frames/')
    dirs = os.listdir(testDirectory)
    dirs = np.sort(dirs).tolist()  # ['000000', '000001', '000002', '000003']
    # dirs.reverse()
    sequenceNumber = -1
    aug = Augmenter()

    if saveBoxes:
        saveBoxesList = []
        saveLabelsList = []

    if isEvaluating:
        for sequence_name in dirs:
            if saveBoxes:
                seqBoxes = []
                seqLabels = []
            sequenceNumber += 1
            if (sequenceNumber - machineID + 1) % machineNumber != 0:
                continue
            if not isReusing:
                print(sequence_name)
            if not sequence_name.endswith('.zip'):
                imageCounter = 0
                imageFiles = os.listdir(os.path.join(testDirectory, sequence_name))  # ['000000.png',...,'005453.png']
                # random.shuffle(imageFiles)
                if isReusing:
                    # print(len(boxList))  # (4,)
                    imageFiles = boxList[sequenceNumber]  # shape:(5201,), imageFiles[0]: box lists:(88,5)(75,5)...
                for image_file in imageFiles:  # (88, 5),(75, 5),(73, 5)(94,5)(80,5)...
                    if isReusing:
                        boxes = boxList[sequenceNumber][imageCounter]  # (88,5),(75,5),(73,5)(94,5)(80,5)...
                        labels = labelList[sequenceNumber][imageCounter]  # (88,)(75,)...
                    else:
                        image_file = os.path.join(testDirectory, sequence_name, image_file)
                        if usingAGC:
                            pilImage = Image.open(image_file).convert('RGB')
                            imArray = np.asarray(pilImage)
                            image_file = aug.augment(imArray, augmentation=Augmentation.AGCWD).astype(np.float)
                        if not usingDropout:
                            detections = inference_detector(model, image_file)
                            if usingGrid:
                                boxes, labels = getBoxesLabels(detections)
                            else:
                                boxes, labels = getBoxesLabels(detections[0])
                        if saveExamples:
                            im = Image.fromarray(image_file.astype(np.uint8))
                            im.save(open('ex.png', 'wb'))
                            img = model.show_result('ex.png', detections, score_thr=0.3, show=False)
                            img = Image.fromarray(img)
                            img.save(open('exout.png', 'wb'))
                            # pdb.set_trace()
                        if usingDropout:
                            boxes = []
                            labels = []
                            for i in range(numDropoutPasses):
                                detections = inference_detector(model, image_file)
                                b, l = getBoxesLabels(detections[0]) if not usingGrid else getBoxesLabels(detections)
                                boxes += b
                                labels += l
                        if usingEnsembles:
                            detections2 = inference_detector(model2, image_file)
                            boxes2, labels2 = getBoxesLabels(detections2)
                            # if usingDropout:
                            #     boxes2 = []
                            #     labels2 = []
                            #     for i in range(numDropoutPasses):
                            #         detections2 = inference_detector(model2, image_file)
                            #         b, l = getBoxesLabels(detections2)
                            #         boxes2 += b
                            #         labels2 += l
                            boxes = boxes + boxes2
                            labels = labels + labels2
                        if saveBoxes:
                            seqBoxes.append(boxes)
                            seqLabels.append(labels)

                    if not cascadeNMS:
                        boxes, labels, var_list, upper_cov_list, lower_cov_list = merge_detections(boxes, labels,
                                                                                                   iouThreshold,
                                                                                                   same_labels=sameLabels)
                        #print(lower_cov_list)
                        # boxes: (14, 5)(12, 5)(14, 5)(16, 5)...
                        # labels: (14,)(12,)(14,)(16,)
                        # var_list: (14, 4),(12, 4)...
                        # pdb.set_trace()
                    else:
                        boxes, labels, var_list = merge_detections(boxes, labels, .9, same_labels=sameLabels)
                        boxes, labels, var_list = merge_detections(boxes, labels, .7, same_labels=sameLabels)
                        boxes, labels, var_list = merge_detections(boxes, labels, .5, same_labels=sameLabels)
                        boxes, labels, var_list = merge_detections(boxes, labels, .3, same_labels=sameLabels)
                        boxes, labels, var_list = merge_detections(boxes, labels, .2, same_labels=sameLabels)
                        boxes, labels, var_list = merge_detections(boxes, labels, .05, same_labels=sameLabels)

                    index = 0
                    labelIndex = -1
                    for box in boxes:  # each 'boxes' includes final bounding boxes in each image
                        labelIndex += 1
                        label = labels[labelIndex]
                        if label not in classids: continue
                        label = classids.index(label)
                        score = box[-1]
                        if removeLowScores:
                            if score < threshold: continue
                        if inflateConfidence:
                            score = 1
                        # if score > .85: score = probability
                        restScore = 1 - score
                        otherScore = restScore / 30
                        scores = [score if idx == label else otherScore for idx in range(len(classes))]
                        # scores = addSingleNone(scores, threshold, probability)
                        xmin, ymin, xmax, ymax = box[:-1]
                        if reduceBoxSize:
                            width = xmax - xmin
                            height = ymax - ymin
                            reductionAmount = boxRatio
                            reducedWidth = reductionAmount * width / 2
                            reducedHeight = reductionAmount * height / 2
                            xmin = xmin + reducedWidth
                            xmax = xmax - reducedWidth
                            ymin = ymin + reducedHeight
                            ymax = ymax - reducedHeight

                        percent = covPercent
                        if useCovPercent:
                            xvar = percent * (xmax - xmin)
                            yvar = percent * (ymax - ymin)

                        if ensVars:
                            xmvar, ymvar, xmxvar, ymxvar = var_list[index]
                            upper_left_cov = make_simple_covariance(clamp(xmvar), clamp(ymvar))
                            lower_right_cov = make_simple_covariance(clamp(xmxvar), clamp(ymxvar))
                        else:
                            upper_left_cov, lower_right_cov = [make_simple_covariance(clamp(xvar), clamp(yvar))
                                                               for i in range(2)]

                            # covariance matrix calculation form SR paper
                            # upper_left_cov, lower_right_cov = upper_cov_list[index], lower_cov_list[index]
                            # if np.array(upper_left_cov).any() == 0:
                            #     upper_left_cov = make_simple_covariance(clamp(xvar), clamp(yvar))
                            # if np.array(lower_right_cov).any() == 0:
                            #     lower_right_cov = make_simple_covariance(clamp(xvar), clamp(yvar))
                            # if not is_positive_definite(upper_left_cov):
                            #     upper_left_cov = find_nearest_positive(np.array(upper_left_cov))
                            # if not is_positive_definite(lower_right_cov):
                            #     lower_right_cov = find_nearest_positive(np.array(lower_right_cov))

                        # # upper_left_cov = [[21890.6578, -5407.4120], [-5407.4120,  5411.8063]]

                        writer.add_detection(scores, xmin, ymin, xmax, ymax, upper_left_cov, lower_right_cov)
                        index += 1


                    """
                        boxes, scores, covariances = mergedDetections
                        index = 0
                        for box in boxes:
                            score = scores[index].tolist()
                            # scores = addSingleNone(scores, threshold, probability)
                            xmin, ymin, xmax, ymax = box[:-1]
                            upLeft, bottomRight = covariances[index]
                            writer.add_detection(score, xmin, ymin, xmax, ymax, upLeft, bottomRight)
        
                            index += 1
                        """
                    writer.next_image()
                    #pdb.set_trace()
                    imageCounter += 1
                    if imageCounter % (numFrames // 2) == 0 and not isSubmitting and not isReusing:
                        print(imageCounter)
                    if imageCounter > numFrames and not isSubmitting:
                        break
                if saveBoxes:
                    saveBoxesList.append(seqBoxes)
                    saveLabelsList.append(seqLabels)
                writer.save_sequence(sequence_name)

    if saveBoxes:
        pickle.dump(saveBoxesList, open(boxFileName, 'wb'))
        pickle.dump(saveLabelsList, open(labelFileName, 'wb'))

    from subprocess import call
    resultNumber = resultNumber
    if not isSubmitting:
        from datetime import datetime
        # datetime object containing current date and time now = datetime.now() dd/mm/YY H:M:S dt_string =
        # now.strftime("%d %m %Y %H %M %S") call(['python', 'rvchallenge-evaluation-master\evaluate.py',
        # '--gt_folder', 'rvchallenge-evaluation-master\ground_truth', '--det_folder',
        # 'rvchallenge-evaluation-master\det', '--save_folder', 'rvchallenge-evaluation-master\\results' + str(
        # resultNumber) + 'per'+ str(covPercent)  + 'ratio'+ str(boxRatio) + 'pro' + str(probability) + 'thr' + str(
        # threshold) + 'numFr' + str(numFrames) + 'date' + dt_string,  '--num_frames', str(numFrames),
        # '--start_frame', '0',])
        call(['python', 'rvchallenge-evaluation-master\evaluate.py', '--gt_folder',
              'rvchallenge-evaluation-master\ground_truth', '--det_folder', 'rvchallenge-evaluation-master\det',
              '--save_folder', 'results', '--num_frames', str(numFrames), '--start_frame', '0', ])
        infile = open('results/scores.txt', 'r')
        filecontent = infile.readlines()
        infile.close()
        array1 = []
        array2 = []
        results = {}
        for line in filecontent:
            tmp = line.strip().split(':')
            results[tmp[0]] = tmp[1]
        return results


def getFileName(usingAGC, usingDropout, usingEnsembles, usingGrid):
    if usingAGC:
        if usingEnsembles:
            boxFileName = 'boxEnsembleAGC.pk'
            labelFileName = 'labelEnsembleAGC.pk'
        else:
            boxFileName = 'boxSingleAGC.pk'
            labelFileName = 'labelSingleAGC.pk'
    else:
        if usingEnsembles:
            boxFileName = 'boxEnsemble.pk'
            labelFileName = 'labelEnsemble.pk'
        else:
            boxFileName = 'boxSingle.pk'
            labelFileName = 'labelSingle.pk'
    if usingDropout:
        boxFileName = 'DO' + boxFileName
        labelFileName = 'DO' + labelFileName
    if usingGrid:
        boxFileName = 'Grid' + boxFileName
        labelFileName = 'Grid' + labelFileName
        # if usingDropout:
        #     boxFileName = 'DO' + boxFileName
        #     labelFileName = 'DO' + labelFileName
    boxFileName = 'savedOutputs/' + boxFileName
    labelFileName = 'savedOutputs/' + labelFileName

    print(boxFileName)
    return boxFileName, labelFileName


# ensemble
# covPercents = [0.2, 0.25, 0.29, 0.3, 0.304, 0.305, 0.31, 0.32,0.33, 0.34, 0.35, 0.37, 0.38, 0.39, 0.4, 0.41,
#                0.42, 0.43, 0.44, 0.45, 0.47, 0.5, 0.55]  # 0.45 # .20
# boxratios = [0.01, 0.05, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.16]   # 0.12,0.13 # .15
# probabilities = [1, ] # doesn't matter
# thresholds = [0.018, 0.02, 0.03, 0.05, 0.08, .1, .2, .3, .4]  # 0.02 # .35
# iouthresholds = [0.1, 0.2, 0.25, 0.28, 0.3, 0.32, 0.4, 0.5, 0.6, 0.8] # 0.3

# covPercents = [0.305]  # .20
# boxratios = [0.1]  # .15
# probabilities = [1, ]
# thresholds = [0.018]  # .35
# iouthresholds = [0.28]

# ensemble + envar
# covPercents = [0.305]  # 0.3
# boxratios = [0.1]  # 0.1, 0.2
# probabilities = [1, ]
# thresholds = [0.018]  # 0.018,  0.2
# iouthresholds = [0.35]  # 0.28, 0.3, 0.5, 0.6

# # dropout
# covPercents = [0.3]  # .2, 0.3
# boxratios = [0.1]  # 0.05, 0.1, 0.2
# probabilities = [1, ]
# thresholds = [0.022]  # 0.022, 0.028, 0.029, 0.03, 0.035, 0.04, 0.05, 0.1, 0.2
# iouthresholds = [0.3]  # 0.15, 0.25, 0.29, 0.3, 0.31, 0.35, 0.4, 0.5 # 0.3

# single htc w/o dropout
# covPercents = [0.305]  # 0.3
# boxratios = [0.1]  #
# probabilities = [1, ]
# thresholds = [0.018,0.022]  # 0.022,  0.2
# iouthresholds = [0.28, 0.3]  #

# dropout
covPercents = [0.3]  # .2, 0.22, 0.25, 0.3, 0.305
boxratios = [0.1]  # 0.05, 0.1, 0.2
probabilities = [1, ]
thresholds = [0.022]  # 0.022, 0.028, 0.029, 0.03, 0.035, 0.04, 0.05, 0.1, 0.2
iouthresholds = [0.28]  # 0.15, 0.2, 0.25, 0.28, 0.29, 0.3, 0.31, 0.35, 0.4, 0.5 # 0.3

bestScore = 0
print('starting tests')

usingDropout = True
usingEnsembles = True
usingAGC = False
isReusing = True
saveBoxes = False
usingGrid = False

if isReusing:
    boxFileName, labelFileName = getFileName(usingAGC, usingDropout, usingEnsembles, usingGrid)
    import pickle

    boxList = pickle.load(open(boxFileName, 'rb'))  # (4, 5201)
    labelList = pickle.load(open(labelFileName, 'rb'))  # (4, 5201)
else:
    boxList, labelList = None, None

# make files
# results = test(covPercent=.3,
#                boxRatio=.1,
#                threshold=0.022,
#                iouThreshold=.28,
#                usingEnsembles=usingEnsembles,
#                usingDropout=usingDropout,
#                usingAGC=usingAGC,
#                boxList=boxList,
#                labelList=labelList,
#                cascadeNMS=False,
#                numFrames=5200,
#                inflateConfidence=True,
#                saveBoxes=saveBoxes,
#                isReusing=isReusing,
#                numDropoutPasses=3,
#                dropoutRate=.3,
#                ensVars=False,
#                sameLabels=False,
#                usingGrid=usingGrid)
# print(results)
# exit(0)

for covPercent in covPercents:
    for boxratio in boxratios:
        for probability in probabilities:
            for threshold in thresholds:
                for iouthreshold in iouthresholds:

                    print('covPercent:', covPercent,
                          ' boxratio:', boxratio,
                          ' probability:', probability,
                          ' threshold:', threshold,
                          ' iouthreshold:', iouthreshold,
                          )
                    results = test(covPercent=covPercent,
                                   boxRatio=boxratio,
                                   probability=probability,
                                   threshold=threshold,
                                   iouThreshold=iouthreshold,
                                   usingEnsembles=usingEnsembles,
                                   usingDropout=usingDropout,
                                   usingAGC=usingAGC,
                                   saveBoxes=saveBoxes,
                                   isReusing=isReusing,
                                   boxList=boxList,
                                   labelList=labelList,
                                   cascadeNMS=False,
                                   numFrames=5200,
                                   usingGrid=usingGrid,
                                   ensVars=False
                                   )
                    score = float(results['score'])
                    print(results)
                    if score > bestScore:
                        print('score improved from ', bestScore, ' to ', score)
                        bestScore = score
