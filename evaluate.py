import os
import numpy as np
import pandas as pd
import csv
import tqdm
from vgg import get_cls,get_img_paths,svm_classify
from vgg import face_verification
from sklearn.metrics import DetCurveDisplay,det_curve
from sklearn.metrics import RocCurveDisplay,roc_curve

def face_identification_eval():
    cls = get_cls()
    train_image_paths,test_image_paths,train_labels,test_gt_labels = get_img_paths(cls)
    test_pred_labels = svm_classify(train_image_paths,train_labels,test_image_paths)
    return test_gt_labels, test_pred_labels

if __name__=='__main__':
    print('='*50)
    print('Reading meta.....')
    print('='*50)
    
    label_path = './test_list.txt'
    score_path = 'scores'
    r_idx_path = 'identity_meta.csv'
    
    # build class to name mapping
    mapping = dict()
    with open('identity_meta.csv','r',encoding='utf-8') as f:
        csvFile = csv.reader(f)
        for line in csvFile:
            mapping[line[0]]=line[1]
    
    if not os.path.exists(score_path):
        os.makedirs(score_path)
        
    # eval identification
    test_gt , test_pred = face_identification_eval()
    tp = 0
    fp = 0
    for i,l in enumerate(test_gt):
        if test_pred[i]==test_gt[i]:
            tp+=1
        else:
            fp+=1
    top_1_err = fp/len(test_gt)
    fpr,fnr,_ = det_curve(test_gt,test_pred)
    det = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name='SVC')
    
        
    