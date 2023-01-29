import threading
import numpy as np
import imutils
from imutils.video import VideoStream
import cv2
import time
from webcamthread import WebcamVideoStream
import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import time
import math
REDU = 8


c = threading.Condition()
##cap = WebcamVideoStream(src = 0).start()
cap = cv2.VideoCapture('D:/Python/yolov5/Video/NewArrow/NewArrow.mp4')
cap1 = cv2.VideoCapture('D:/Python/yolov5/Video/NewArrow/NewArrow.mp4')
time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Navigation.avi',fourcc, 35.0, (320 , 240))
ls = []
def calc_bound(ls):
       if ls[0][0] < ls[1][0] and ls[0][1] < ls[1][1] :   # top left
                  x1 = ls[0][0]
                  y1 = ls[0][1]
                  x2 = ls[1][2]
                  y2 = ls[1][3]
       elif ls[0][0] < ls[1][0] and ls[0][1] > ls[1][1] :   # bottom left
                  x1 = ls[0][0]
                  y1 = ls[0][3]
                  x2 = ls[1][2]
                  y2 = ls[1][1]
       elif ls[0][0] > ls[1][0] and ls[0][1] > ls[1][1] :   # bottom right
                  x1 = ls[1][0]
                  y1 = ls[1][1]
                  x2 = ls[0][2]
                  y2 = ls[0][3]
       elif ls[0][0] > ls[1][0] and ls[0][1] < ls[1][1] :   # Top right
                  x1 = ls[1][0]
                  y1 = ls[0][1]
                  x2 = ls[0][2]
                  y2 = ls[1][3]
       else :
                  x1 = ls[0][0]
                  y1 = ls[0][1]
                  x2 = ls[1][2]
                  y2 = ls[1][3]
       return (x1,y1) , (x2,y2)

def angle(ls):
    if((ls[0][1]-ls[1][1])<0 and (ls[0][0]-ls[1][0])>0):
        if((ls[0][0]-ls[1][0])!=0):
            slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
            theta=math.degrees(math.atan(slope))
            theta=-theta
        else:
            theta=90
        
    elif((ls[0][0]-ls[1][0])<0 and (ls[0][1]-ls[1][1])>0):
        if((ls[0][0]-ls[1][0])!=0):
            slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
            theta=math.degrees(math.atan(slope))
            theta=180-theta 
        else:
            theta=270
        
    elif((ls[0][0]-ls[1][0])<0 and (ls[0][1]-ls[1][1])<0):
        if((ls[0][0]-ls[1][0])!=0):
            slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
            theta=math.degrees(math.atan(slope))
            theta=180-theta
        else:
            theta=90
        
    else:
        if((ls[0][0]-ls[1][0])!=0):
            slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
            theta=math.degrees(math.atan(slope))
            theta=360-theta
        else:
            theta=270
    return theta


class DNNThread(threading.Thread):
          
           def __init__(self , opt,tobj):
                      threading.Thread.__init__(self)
                      self.out, self.source, self.weights, self.view_img, self.save_txt, self.imgsz = \
                              opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size 
                      set_logging()
                      self.device = select_device(opt.device)
                      self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
                      self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())
##                      self.cudnn.benchmark = True
                      self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                      self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
                      img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
                      _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once
                     
                      self.opt = opt
                      self.stopit = False
                      self.out = False 
                      self.g = False
                      print('fkdnvsldj')
                      self.t=tobj
                      self.head=False
                      self.tail=False
                      self.proc=False
                      self.detFlag=False
                      self.dhead=True
                      self.dtail=True
                      self.count=0
                      self.runcode()
                      
           def getm(self):
                      return self.m
           def run(self):
                      global cap
                      
                      try :
                                 while not self.stopit:
                                            self.runcode()
##                                            time.sleep(0.1)

                      except Exception as e:
                                 print('Error occured in DNN Thread\n' , e)
                                 #self.out = True
                                 self.run()

           def runcode(self):
                      e = True
                      self.count = 0
                      #count=0
                      total = 0
                      initial=False
                      global ls
                      ls1=[[0,0,0,0],[0,0,0,0]]
                      while e :
                                 t1 = time.time()
                                 self.img=self.t.getframe()
                                 self.count += 1
                                 initial=True 
                                 self.img = cv2.resize(self.img , (640, 380))
                                 
                                 img0 = letterbox(self.img , new_shape = self.imgsz)[0]
                                 img0 = img0[:, :, ::-1].transpose(2, 0, 1)
                                 img0 = np.ascontiguousarray(img0)
                                 img0 = torch.from_numpy(img0).to(self.device).float()
                                 img0 /= 255.0
                                 if img0.ndimension() == 3:
                                             img0 = img0.unsqueeze(0)

                                 pred = self.model(img0, augment=self.opt.augment)[0]
                                 pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
                                 lis=[]
                                 ls=[]
                                 
                                 for i , det in enumerate(pred):
                                            gn = torch.tensor(self.img.shape)[[1, 0, 1, 0]]
                                            
                                            
                                            if det is not None and len(det):
                                                 det[: , : 8] = scale_coords(img0.shape[2:] , det[: , :8] , self.img.shape).round()
                                                 
                                                 for c in det[:,-1].unique():
                                                            n = (det[:,-1] == c).sum()
                                                 
                                                 for *xyxy , conf , cls in reversed(det) :
                                                       bbox = []
                                                       label = '%s %.2f' % (self.names[int(cls)], conf)
                                                       
                                                       plot_one_box(xyxy, self.img, label=label, color=self.colors[int(cls)], line_thickness=3)
                                                       if int(xyxy[2].item())>self.img.shape[1] or int(xyxy[3].item()) >self.img.shape[0] or int(xyxy[0].item()) < 0 or int(xyxy[1].item()) < 0:
                                                            break
                                                        #bbox = [int(xyxy[0].item()) , int(xyxy[1].item()) , int(xyxy[2].item() - xyxy[0].item()) , int(xyxy[3].item() - xyxy[1].item()) ]
                                                       #print("Done1")
                                                       print(label)
                                                       lis.append(label)
                                                       ls.append([int(xyxy[0].item()) , int(xyxy[1].item()),int(xyxy[2].item()) , int(xyxy[3].item())])
                                                       #print("Done2")
                                                       self.proc=True
                                                       #print("ls det",len(ls))
                                                          
                                                                                                            

                                                           #print(self.detFlag)
                                                           #print("Detect")
                                                       cv2.putText(self.img , "Detecting" , (100 , 140) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       self.proc=False
                                                       #lis.append(bbox)     
##                                                       bbox = np.array(bbox)
                                                       
                                                 e = False
                                 

                                 if(len(ls)==2):
                                        if(lis[1]=="head 1.00"):
                                               ls.append(ls[0])
                                               ls.remove(ls[0])
                                               lis.append(lis[0])
                                               lis.remove(lis[0])
                                               
                                 #print(ls,lis)
                                 if(len(ls)==2 and lis[0]=="head 1.00"):
                                     bbox=[int(ls[0][0]) , int(ls[0][1]) , int(ls[0][2] - ls[0][0]) , int(ls[0][3] - ls[0][1])]
                                     while(self.head):
                                          continue
                                     self.trackerhead = cv2.TrackerKCF_create()
                                     frame1=self.t.getframe()
                                     frame1=cv2.resize(frame1 , (640, 380))
                                     self.dhead=True
                                     ok = self.trackerhead.init(frame1 , tuple(bbox))

                                 self.dhead=False
                                 #print("Done3")
                               
                                 if(len(ls)==2 and lis[1]=="tail 1.00"):
                                     bbox=[int(ls[1][0]) , int(ls[1][1]) , int(ls[1][2] - ls[1][0]) , int(ls[1][3] - ls[1][1])]
                                     while(self.tail):
                                         continue
                                     self.trackertail = cv2.TrackerKCF_create()
                                     frame1=self.t.getframe()
                                     frame1=cv2.resize(frame1 , (640, 380))
                                     self.dtail=True
                                     ok = self.trackertail.init(frame1 , tuple(bbox))

                                 self.dtail=False
                                 
                                 #cv2.imshow('frame' , self.img)
                                 #cv2.waitKey(10)
                                 #count += 1
                                 total =total  + (time.time() - t1)
                                 #print("ls1 ",len(ls1))
                                 #print("Detect")
                                 #print(total / self.count)
                                
                                 

class TrackerThread(threading.Thread):
           def __init__(self , opt):
                      threading.Thread.__init__(self)
                      ret,self.frame=cap.read()
                      self.b = DNNThread(opt,self)
                      self.que1=[]
                      self.b.start()
                      
           def getframe(self):
                  
                  #print(self.c)
                  return self.frame
                      
           def run(self):
                      
                      global out
                      
                      bgsub = cv2.createBackgroundSubtractorMOG2(500, 300, True)
                      kernel = np.ones((3,3),np.uint8)
                      count = 0
                      tim=1
                      try :
                                 while True:
                                                   
                                            ret , frame = cap.read()
                                            self.frame=frame
                                            frame = cv2.resize(frame , (640, 380))
                                            bgs = bgsub.apply(frame)
                                            bgs = cv2.erode(bgs,kernel,iterations = 1)
                                            bgs = cv2.medianBlur(bgs,3)
                                            bgs = cv2.dilate(bgs,kernel,iterations=2)
                                            bgs = (bgs > 200).astype(np.uint8)*255
                                            print("Adi")
                                            cv2.imshow("Img1",frame)
                                            key = cv2.waitKey(1)
                                            if key == 27:
                                                   break
                                            
                                            ts=[]
                                            self.b.head=True

                                            
                                            ok , bbox =  self.b.trackerhead.update(frame)
                                            if ok:

                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break

                                                       if(self.b.proc==False):
                                                              ts.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 0 , 0 , 255 ) , 2 , 1)
                                            else :
                                                       cv2.putText(frame , "Tracking failure detected in HEAD" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       
                                                       pass
                                            
                                            self.b.head=False
                                            

                                            

                                            self.b.tail=True

                                            ok , bbox =  self.b.trackertail.update(frame)
                                            if ok:
                                                       #print(6)
                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break

                                                       if(self.b.proc==False):

                                                              ts.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 255 , 0 , 0 ) , 2 , 1)
                                            else :
                                                       cv2.putText(frame , "Tracking failure detected in TAIL" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       pass
                                            self.b.tail=False

                                            if(len(ts)==2):
                                                self.que1.append((ts,tim))
                                                tim=1
                                                cv2.putText(frame , "Angle is "+str(angle(ts))+"degrees" , (100 ,300) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 0) , 2)
                                                print(angle(ts))
                                            elif(len(ts)==1):
                                                ts.remove(ts[0])
                                                tim+=1

                                            elif(len(ts)==0):
                                                tim+=1
                                                
                                            cv2.imshow('frame1' , frame)
                                            
                                            self.b.count+=1

                                            if  cv2.waitKey(10) & 0xff  == 27 or self.b.out :
                                                        self.getout()
                                                        break
                                            #time.sleep(0.53)
                                            #time.sleep(1/60.05)
                                            #print("Track")
                      except Exception as e:
                                             print('Error occured in Tracker thread \n', e )
                                             #self.getout()
                                             self.run()
           def getout(self):
                      print('Exiting the program')
                      self.b.stopit = True
                      self.b.join()
                      #cap.stop()
                      out.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/Python/yolov5/NewYolo/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/Python/yolov5/Video/NewArrow/NewArrow.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='C:/Users/Admin/yolo_training/inference/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
##    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    a = TrackerThread(opt)
    a.start()
    a.join()
    
    print('Finished....')
