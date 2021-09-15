import os
import cv2
import glob
import random
import numpy as np
from mtcnn.mtcnn import MTCNN

def extract_opencv(filename):
    detector = MTCNN()
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            faces = detector.detect_faces(frame)
            if(len(faces)!=0):
                video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]


basedir = '/home/ed716/Documents/NewSSD/lrs2/pretrain'
basedir_to_save = '/home/ed716/Documents/NewSSD/Cocktail/face'
filenames = glob.glob(os.path.join(basedir, '*', '*.mp4'))
for filename in filenames:
    data = extract_opencv(filename)  #(n_frame, 160, 160, 3)
    if data.ndim == 4:
        data = data[0, :, :, :]
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        path_to_save = os.path.join(basedir_to_save,
                                    filename.split('/')[-2]+'_'+
                                    filename.split('/')[-1][:-4]+'.jpg')
        cv2.imwrite(path_to_save, data)
