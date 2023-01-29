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
from kalman import kalman, ukf
import htrans as ht
import time
REDU = 8


c = threading.Condition()
##cap = WebcamVideoStream(src = 0).start()
cap = cv2.VideoCapture('D:/Python/yolov5/aa.mp4')
#time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Navigation.avi',fourcc, 35.0, (320 , 240))

##########          kalman filter matrices        ################





class DNNThread(threading.Thread):
          
           def __init__(self , opt):
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
                      self.head=True
                      self.tail=True
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
                                 self.out = True

           def runcode(self):
                      e = True
                      count = 0
                      total = 0
                      
                      while (self.head==True and self.tail==True) :
                                 t1 = time.time()
                                 ret , self.img = cap.read()
                                 count += 1
                                 #print(ret)
                                 '''
                                 if count < 20:
                                            #print(count)
                                            continue
                                 '''
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
                                 
                                 for i , det in enumerate(pred):
                                            gn = torch.tensor(self.img.shape)[[1, 0, 1, 0]]
                                            
                                            lis=[]
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
                                                       bbox = [int(xyxy[0].item()) , int(xyxy[1].item()) , int(xyxy[2].item() - xyxy[0].item()) , int(xyxy[3].item() - xyxy[1].item()) ]
                                                       #lis.append(bbox)     
##                                                       bbox = np.array(bbox)
                                                       if(str(label)=="head 1.00" and self.head==True):
                                                                  self.trackerhead = cv2.TrackerKCF_create()
                                                                  ok = self.trackerhead.init(self.img , tuple(bbox))
                                                                  self.head=False
                                                       elif(str(label)=="tail 1.00" and self.tail==True):
                                                                  self.trackertail = cv2.TrackerKCF_create()
                                                                  ok = self.trackertail.init(self.img , tuple(bbox))
                                                                  self.tail=False
                                                       e = False
                                 
                                 cv2.imshow('frame' , self.img)
                                 cv2.waitKey(10)
                                 count += 1
                                 total =total  + (time.time() - t1)
                                 
                                 #print(total / count)
                                
                                 

class TrackerThread(threading.Thread):
           def __init__(self , opt):
                      threading.Thread.__init__(self)
                      self.b = DNNThread(opt)
                      self.b.start()
                      
                      self.keyh=False
                      self.keyt=False
           def run(self):
                      global out
                      ''' 
                      degree = np.pi/180
                      a = np.array([0, 900])

                      #fps = 60
                      fps = 28
                      dt = 1/fps
                      t = np.arange(0,2.01,dt)
                      noise = 5

                      F = np.array(
                           [1, 0, dt, 0,
                           0, 1, 0, dt,
                           0, 0, 1, 0,
                           0, 0, 0, 1 ]).reshape(4,4)
                      B = np.array(
                           [dt**2/2, 0,
                           0, dt**2/2,
                           dt, 0,
                           0, dt ]).reshape(4,2)

                      H = np.array(
                            [1,0,0,0,
                           0,1,0,0]).reshape(2,4)

                      # x, y, vx, vy
                      mu = np.array([0,0,0,0])
                      # your uncertainties
                      P = np.diag([1000,1000,1000,1000])**2
                      print('P = ', P)
                      #res = [(mu,P,mu)]
                      res=[]
                      N = 15 # to take an initial section and see what happens if it is later lost
                      sigmaM = 0.0001 # model noise
                      sigmaZ = 3 * noise # should be equal to the average noise of the image process. 10 pixels pje.
                      Q = sigmaM**2 * np.eye(4)
                      R = sigmaZ**2 * np.eye(2)

                      listCenterX=[]
                      listCenterY=[]
                      listpuntos=[]
                      '''
                      bgsub = cv2.createBackgroundSubtractorMOG2(500, 300, True)
                      kernel = np.ones((3,3),np.uint8)
                      count = 0
                      
                      try :
                                 while True:
                                            ret , frame = cap.read()
                                            frame = cv2.resize(frame , (640, 380))
                                            bgs = bgsub.apply(frame)
                                            bgs = cv2.erode(bgs,kernel,iterations = 1)
                                            bgs = cv2.medianBlur(bgs,3)
                                            bgs = cv2.dilate(bgs,kernel,iterations=2)
                                            bgs = (bgs > 200).astype(np.uint8)*255
                                            
                                            ls=[]
                                            
                                            
                                            #if(self.b.head==True):
                                                       
                                                       #self.b.head=False
                                            ok , bbox =  self.b.trackerhead.update(frame)
                                            if ok:
                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break
                                                       ls.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 0 , 0 , 255 ) , 2 , 1)
                                                       #print("IN HEAD")
                                            else :
                                                       #cv2.putText(frame , "Tracking failure detected in HEAD" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       #self.b.head=True
                                                       self.keyh=True
                                                       #print(2)
                                            #if(self.b.tail==True):
                                                       #self.b.tail==False
                                            ok , bbox =  self.b.trackertail.update(frame)
                                            if ok:
                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break
                                                       ls.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 255 , 0 , 0 ) , 2 , 1)
                                                       #print("IN TAIL")
                                            else :
                                                       #cv2.putText(frame , "Tracking failure detected in TAIL" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       #self.b.tail=True
                                                       self.keyt=True
                                                       #print(3)
                                                       
                                            if(self.keyh and self.keyt):
                                                self.keyh=False
                                                self.keyt=False
                                                count=0
                                                print("0")
                                                self.b.head=True
                                                self.b.tail=True
                                                       
                                            
                                                '''
                                                       if ok:
                                                                  p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                                  p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                                  if p1[0] < 0 or p1[1] <0 :
                                                                             print('out of bound')
                                                                             print('skipping frames')
                                                                             self.trackMode = False
                                                                             break
                                                                  ls.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                                  cv2.rectangle(frame , p1 , p2 , ( 255 , 0 , 0 ) , 2 , 1)
                                                       else :
                                                                  cv2.putText(frame , "Tracking failure detected" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                '''

                                            ##########Kalman filter############
                                            '''
                                            xo=int(bbox[0]+bbox[2]/2)
                                            yo=int(bbox[1]+bbox[3]/2)
                                            error=(bbox[3])
                                            #Roibix center calculations
                                            #print(yo,error)
                                           
                                            if(yo<error  or bgs.sum()<50 ):
                                                 mu,P,pred= kalman(mu,P,F,Q,B,a,None,H,R)
                                                 m="None"
                                                 mm=False
                                            else:
                                                mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
                                                m="normal"
                                                mm=True
                                            if(mm):
                                                 listCenterX.append(xo)
                                                 listCenterY.append(yo)
 
                                                 listpuntos.append((xo,yo,m))
                                                 res += [(mu,P)]
                                             ##### Prediccion #####
                                            mu2 = mu
                                            P2 = P
                                            res2 = []
                                            for _ in range(fps*2):
                                                  mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
                                                  res2 += [(mu2,P2)]

                                            xe = [mu[0] for mu,_ in res]
                                            xu = [2*np.sqrt(P[0,0]) for _,P in res]
                                            ye = [mu[1] for mu,_ in res]
                                            yu = [2*np.sqrt(P[1,1]) for _,P in res]
                                            xp=[mu2[0] for mu2,_ in res2]
                                            yp=[mu2[1] for mu2,_ in res2]
                                            xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
                                            ypu = [2*np.sqrt(P[1,1]) for _,P in res2]
 
                                            for n in range(len(listCenterX)): # centro del roibox
                                                 cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,
                                                 (0, 255, 0),-1)

                                            for n in range(len(xe)): # xe e ye estimada [-1]
                                            #incertidumbre = (xu[n]*yu[n])
                                            #cv.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),-1)
                                                 incertidumbre=(xu[n]+yu[n])/2
                                                 cv2.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)

                                            for n in range(len(xp)): #Roibix center calculations
                                                 incertidumbreP=(xpu[n]+ypu[n])/2
                                                 cv2.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))
                                                 
                                  ##
                                  ##          print("Lista de puntos\n")
                                  ##          for n in range(len(listpuntos)):
                                  ####               print(listpuntos[n])
                                  ##               pass

                                            if(len(listCenterY)>4):
                                                 if((listCenterY[-5] < listCenterY[-4]) and(listCenterY[-4] < listCenterY[-3]) and (listCenterY[-3] > listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):
                                                      print("REBOTE")
                                                      listCenterY=[]
                                                      listCenterX=[]
                                                      listpuntos=[]
                                                      res=[]
                                                      mu = np.array([0,0,0,0])
                                                      P = np.diag([100,100,100,100])**2
                                                      
                                            '''
                                            count+=1
                                            if(count==10):
                                                count=0
                                                print("1")
                                                self.b.head=True
                                                self.b.tail=True
                                            cv2.imshow('frame1' , frame) 
                                            if  cv2.waitKey(10) & 0xff  == 27 or self.b.out :
                                                        self.getout()
                                                        break
                                            #time.sleep(0.53)
                                            time.sleep(1/28)
                                            #print("Track")
                      except Exception as e:
                                             print('Error occured in Tracker thread \n', e )
                                             self.getout()
           def getout(self):
                      print('Exiting the program')
                      self.b.stopit = True
                      self.b.join()
                      #cap.stop()
                      out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/Python/yolov5/best_without_box.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/Python/yolov5/aa.mp4', help='source')  # file/folder, 0 for webcam
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
