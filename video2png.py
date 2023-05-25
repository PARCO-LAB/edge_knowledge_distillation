

import glob
import sys

import cv2
import os

camera = '55011271'
for subject in ['S1','S5', 'S6', 'S7', 'S8', 'S9', 'S11']:

    files = glob.glob('/home/shared/nas/dataset/Human3.6M/' + subject + '/Videos/*' +camera+'.mp4')
    if not os.path.exists('/home/shared/nas/KnowledgeDistillation/h36m/' + subject):
        os.makedirs('/home/shared/nas/KnowledgeDistillation/h36m/' + subject)

    if not os.path.exists('/home/shared/nas/KnowledgeDistillation/h36m/' + subject + '/' + camera):
        os.makedirs('/home/shared/nas/KnowledgeDistillation/h36m/' + subject + '/' + camera)    
    
    for file in files:
        name = file.split('/')[-1]

        if ' ' in name:
            name = "".join(name.split(' '))
            
        
        name = name.split('.') 
        name = ".".join(name[0:len(name)-1])
        print(name)
        if not os.path.exists('/home/shared/nas/KnowledgeDistillation/h36m/' + subject + '/' + camera +'/' +name):
            os.makedirs('/home/shared/nas/KnowledgeDistillation/h36m/' + subject + '/' + camera +'/' +name)    
        
        vidcap = cv2.VideoCapture(file)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite('/home/shared/nas/KnowledgeDistillation/h36m/' + subject + '/' + camera +'/' +name +"/%d.png" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1
        vidcap.release()