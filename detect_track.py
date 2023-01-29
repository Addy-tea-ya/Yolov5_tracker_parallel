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
import math
import copy
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
##########          kalman filter matrices        ################
class Kal:
    def __init__(self):
        self.xc=0
        self.yc=0
        self.slope=0
        self.sx=0
        self.sy=0
        self.stheta=0
        self.ax=0
        self.ay=0
        self.atheta=0
        self.vx=0
        self.vy=0
        self.vtheta=0
        self.theta=0
        self.dt=1/28
        self.vxa=[]
        self.vya=[]
        self.vthetaA=[]
        self.sxa=[]
        self.sya=[]
        self.sthetaA=[]
        self.sthetaB=[]
        self.fut=[]
        self.futv=[]
        self.ang=0
        self.rev=0
        self.The=0
        self.revf=False
        self.HD=[]
        self.TD=[]
    def angz(self,prev,pres):
        angle=0
       
        if(pres<=180 and pres>=90 and prev<=270 and prev>=180):
            self.revf=True
            #print("A")
        elif(prev<=180 and prev>=90 and pres<=270 and pres>=180):
            self.revf=True
            #print("B")
           
        if(pres<=90 and pres>=0 and prev<=360 and prev>=270 and (self.revf or self.rev>=1)):
            self.rev+=1
            angle=(self.rev*360)+pres
            self.revf=False
            #print("a")
        elif(prev<=90 and prev>=0 and pres<=360 and pres>=270 and (self.revf or self.rev>=1)):
            self.rev-=1
            angle=(self.rev*360)+pres
            self.revf=False
            #print("b")
        else:
            angle=(self.rev*360)+pres
       #print(round(prev,2),"\t",round(pres,2),"\t",round(angle,2),"\t",self.revf,"\t",self.rev,"\n")
       #print(round(angle,2),"\t",self.rev,"\n")
        return angle
    
    def kalm(self,frame,pastim):
                      
                      
                      global ls
                      fps = 28
                      dt = 1/fps
                      t = np.arange(0,2.01,dt)
                      noise = 5
                      self.frame=frame
                      F = np.array(
                      [1, 0, 0, dt,0,0,
                       0, 1, 0, 0,dt,0,
                       0, 0, 1, 0,0,dt,
                       0, 0, 0, 1,0,0,
                       0,0,0,0,1,0,
                       0,0,0,0,0,1]).reshape(6,6)
                      B = np.array(
                       [dt**2/2, 0,0,
                        0, dt**2/2,0,
                        0,0,dt**2/2,
                        dt, 0,0,
                       0, dt,0,
                        0,0,dt]).reshape(6,3)

                      H = np.array(
                       [1,0,0,0,0,0,
                       0,1,0,0,0,0,
                        0,0,1,0,0,0,
                        0,0,0,1,0,0,
                        0,0,0,0,1,0,
                        0,0,0,0,0,1]).reshape(6,6)

# x, y, vx, vy
                      mu = np.array([0,0,0,0,0,0])
                      mu1 = np.array([0,0,0,0,0,0])
# your uncertainties  
                      P = np.diag([10000,10000,10000,10000,10000,10000])**2
                      P1 = np.diag([10000,10000,10000,10000,10000,10000])**2
#res = [(mu,P,mu)]
                      res=[]
                      N = 15 # to take an initial section and see what happens if it is later lost
                      sigmaM = 0.0001 # model noise
                      sigmaZ = 3 * noise # should be equal to the average noise of the image process. 10 pixels pje.
                      Q = sigmaM**2 * np.eye(6)
                      R = sigmaZ**2 * np.eye(6)
                      F1= np.array(
                           [1, 0, self.dt, 0,
                           0, 1, 0, self.dt,
                           0, 0, 1, 0,
                           0, 0, 0, 1 ]).reshape(4,4)
                      B1 = np.array(
                           [self.dt**2/2, 0,
                           0, self.dt**2/2,
                           self.dt, 0,
                           0, self.dt ]).reshape(4,2)

                      H1 = np.array(
                            [1,0,0,0,
                           0,1,0,0,
                           0,0,1,0,
                           0,0,0,1]).reshape(4,4)
                      mu21=np.array([0,0,0,0])
                      # x, y, vx, vy
                      mu11 = np.array([0,0,0,0])
                      mu31=np.array([0,0,0,0])
                      pred1 = np.array([0,0,0,0])
                      pred31=np.array([0,0,0,0])
                      # your uncertainties
                      P11 = np.diag([1000,1000,1000,1000])**2
                      P31= np.diag([1000,1000,1000,1000])**2
                      #print('P = ', P)
                      #res = [(mu,P,mu)]
                      res11=[]
                      res31=[]
                      N = 15 # to take an initial section and see what happens if it is later lost
                      sigmaM = 0.0001 # model noise
                      sigmaZ = 3 * noise # should be equal to the average noise of the image process. 10 pixels pje.
                      Q1 = sigmaM**2 * np.eye(4)
                      R1 = sigmaZ**2 * np.eye(4)
                      #a = np.array([200,200,200])  
                      #mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([x1,y1]),H,R)
                      #mu1,P1,pred1= kalman(mu1,P1,F,Q,B,a,np.array([x2,y2]),H,R)
                      while True:
                                 global ls
                                 #print("Length ",len(ls))
                                 if len(ls) == 2 :
                                            #print(ls)
                                            if((ls[0][1]-ls[1][1])<0 and (ls[0][0]-ls[1][0])>0):
                                                if((ls[0][0]-ls[1][0])!=0):
                                                    self.slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
                                                    self.theta=math.degrees(math.atan(self.slope))
                                                    self.theta=-self.theta
                                                else:
                                                    self.theta=90
                                                
                                            elif((ls[0][0]-ls[1][0])<0 and (ls[0][1]-ls[1][1])>0):
                                                if((ls[0][0]-ls[1][0])!=0):
                                                    self.slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
                                                    self.theta=math.degrees(math.atan(self.slope))
                                                    self.theta=180-self.theta 
                                                else:
                                                    self.theta=270
                                                
                                            elif((ls[0][0]-ls[1][0])<0 and (ls[0][1]-ls[1][1])<0):
                                                if((ls[0][0]-ls[1][0])!=0):
                                                    self.slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
                                                    self.theta=math.degrees(math.atan(self.slope))
                                                    self.theta=180-self.theta
                                                else:
                                                    self.theta=90
                                                
                                            else:
                                                if((ls[0][0]-ls[1][0])!=0):
                                                    self.slope=(ls[0][1]-ls[1][1])/(ls[0][0]-ls[1][0])
                                                    self.theta=math.degrees(math.atan(self.slope))
                                                    self.theta=360-self.theta
                                                else:
                                                    self.theta=270
                                            
                                            
                                            #print(self.theta)
                                            bound = calc_bound(ls)
                                            self.xc = bound[0][0]+((bound[1][0]-bound[0][0])/2)
                                            self.yc = bound[0][1]+((bound[1][1]-bound[0][1])/2)
                                            #cv2.circle(self.frame,(int(self.xc),int(self.yc)),5,(0,0,255),1)
                                            
                                            #cv2.rectangle(self.frame , bound[0] , bound[1] , (255, 255, 0) , 2 , 1)
                                            leng=12
                                            mul=1
                                            a=np.array([int(self.ax)*mul,int(self.ay)*mul,int(self.atheta)*mul])
                                            a1=np.array([int(self.ax)*mul,int(self.ay)*mul])
                                            
                                            mu11,P11,pred11= kalman(mu11,P11,F1,Q1,B1,a1,np.array([int(self.xc) ,int(self.yc),int(self.vx)*mul,int(self.vy)*mul]),H1,R1)
                                            for i in range(pastim):##Tatpurta JUGAD
                                                   self.sxa.append(self.xc)
                                                   self.sya.append(self.yc)
                                                   self.sthetaA.append(self.theta)
                                            
                                            headlen=(((ls[0][0]+((ls[0][2]-ls[0][0])/2))-self.xc)**2+((ls[0][1]+((ls[0][3]-ls[0][1])/2))-self.yc)**2)**0.5
                                            #taillen=(((ls[0][0]+((ls[1][0]-ls[0][0])/2)-self.xc)**2+((ls[0][1]+((ls[1][1]-ls[0][1])/2))-self.yc)**2)**0.5
                                            taillen=(((ls[1][0]+((ls[1][2]-ls[1][0])/2))-self.xc)**2+((ls[1][1]+((ls[1][3]-ls[1][1])/2))-self.yc)**2)**0.5
                                            widthh=ls[0][2]-ls[0][0]
                                            heighth=ls[0][3]-ls[0][1]
                                            widtht=ls[1][2]-ls[1][0]
                                            heightt=ls[1][3]-ls[1][1]
                                            angL=0
                                            if(len(self.sxa)==leng):
                                                   self.sx=self.sxa[0]
                                                   self.sxa.remove(self.sxa[0])
                                                   self.vx=(self.xc-self.sx)/((leng-1)*self.dt)
                                                   self.vxa.append(self.vx)
                                            if(len(self.vxa)==leng):
                                                   vx=self.vxa[0]
                                                   self.vxa.remove(self.vxa[0])
                                                   self.ax=(self.vx-vx)/((leng-1)*self.dt)
                                                   
                                            if(len(self.sya)==leng):
                                                   self.sy=self.sya[0]
                                                   self.sya.remove(self.sya[0])
                                                   self.vy=(self.yc-self.sy)/((leng-1)*self.dt)
                                                   self.vya.append(self.vy)
                                            if(len(self.vya)==leng):
                                                   vy=self.vya[0]
                                                   self.vya.remove(self.vya[0])
                                                   self.ay=(self.vy-vy)/((leng-1)*self.dt)
                                                 
                                            if(len(self.sthetaA)==leng):
                                                   self.stheta=self.sthetaA[0]
                                                   self.The=self.sthetaA[leng-2]
                                                   angL=int(self.angz(self.The,self.theta))
                                                   self.sthetaB.append(angL)
                                                   if(len(self.sthetaB)==leng):
                                                       self.vtheta=(angL-self.sthetaB[0])/((leng-1)*self.dt)
                                                       self.sthetaB.remove(self.sthetaB[0])
                                                   self.sthetaA.remove(self.sthetaA[0])
                                                   self.vthetaA.append(self.vtheta)
                                            if(len(self.vthetaA)==leng):
                                                   vtheta=self.vthetaA[0]
                                                   self.vthetaA.remove(self.vthetaA[0])
                                                   self.atheta=(self.vtheta-vtheta)/((leng-1)*self.dt)
                                                   
                                            
                                            mu,P,pred1= kalman(mu,P,F,Q,B,a,np.array([int(self.xc) ,int(self.yc),angL,int(self.vx)*mul,int(self.vy)*mul,int(self.vtheta)*mul]),H,R)
                                            res+=[(mu,P)]
                                            mu2 = np.array([int(self.xc),int(self.yc),angL,mu[3],mu[4],mu[5]])
                                            P2 = P
                                            res2 = []
                                            mu21 = np.array([int(self.xc),int(self.yc),mu[2],mu[3]])
                                            P21 = P11
                                            res21 = []
                                            
                                            for _ in range(4):
                                                  mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
                                                  res2 += [(mu2,P2)]
                                                  mu21,P21,pred21= kalman(mu21,P21,F1,Q1,B1,a1,None,H1,R1)
                                                  res21 += [(mu21,P21)]
                                            self.fut.append(mu2[2])
                                            self.futv.append(mu2[5])
                                            if(len(self.fut)==4):
                                                self.ang=self.fut[0]
                                                #print(angL,"\t",round(self.fut[0],2),"\t")
                                                self.fut.remove(self.fut[0])
                                                self.futv.remove(self.futv[0])
                                            Angl = mu2[2]-self.rev*360
                                            
                                            xh =  mu2[0] + (headlen *math.cos(Angl * math.pi / 180.0))
                                            yh =  mu2[1] - (headlen *math.sin(Angl * math.pi / 180.0))
                                            hea1 =  (int(xh-(widthh/2)),int(yh-(heighth/2)))
                                            hea2 =  (int(xh+(widthh/2)),int(yh+(heighth/2)))
                                            
                                            xt =  mu2[0] + (taillen *math.cos((180+Angl)* math.pi / 180.0))
                                            yt =  mu2[1] - (taillen *math.sin((180+Angl)* math.pi / 180.0))
                                            ta1 =  (int(xt-(widtht/2)),int(yt-(heightt/2)))
                                            ta2 =  (int(xt+(widtht/2)),int(yt+(heightt/2)))
                                            self.HD.append([hea1,hea2])
                                            self.TD.append([ta1,ta2])
                                            #print(self.HD)
                                            if(len(self.HD)==4):
                                                #cv2.rectangle(self.frame, self.HD[0][0], self.HD[0][1], (255,0,255), 2 , 1)
                                                #cv2.rectangle(self.frame, self.TD[0][0], self.TD[0][1], (0,0,0), 2 , 1)
                                                self.HD.remove(self.HD[0])
                                                self.TD.remove(self.TD[0])
                                            #print(Angl,"\t",headlen,"\t",taillen)
                                            xe = [mu[0] for mu,_ in res]
                                            xu = [2*np.sqrt(P[0,0]) for _,P in res]
                                            ye = [mu[1] for mu,_ in res]
                                            yu = [2*np.sqrt(P[1,1]) for _,P in res]
                                            te = [mu[2] for mu,_ in res]
                                            tu = [2*np.sqrt(P[2,2]) for _,P in res]
                                            xp=[mu2[0] for mu2,_ in res2]
                                            yp=[mu2[1] for mu2,_ in res2]
                                            tp=[mu2[2] for mu2,_ in res2]
                                            xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
                                            ypu = [2*np.sqrt(P[1,1]) for _,P in res2]
                                            tpu = [2*np.sqrt(P[2,2]) for _,P in res2]
                                            #print(self.rev)
                                            cv2.circle(self.frame,(int(self.xc),int(self.yc)),10,(255,0,255),4)
                                            cv2.circle(self.frame,(int(mu2[0]),int(mu2[1])),10,(255,0,0),4)
                                            #cv2.circle(self.frame,(int(xh),int(yh)),20,(0,255,255),4)
                                            #cv2.circle(self.frame,(int(xt),int(yt)),20,(0,255,0),4)
                                            #cv2.rectangle(self.frame, hea1, hea2, (0,0,255), 2 , 1)
                                            #cv2.rectangle(self.frame, ta1, ta2, (0,255,0), 2 , 1)
                                            #cv2.circle(self.frame,(int(mu21[0]),int(mu21[1])),10,(255,255,0),4)
                                            #cv2.circle(self.frame,(int(mu11[0]),int(mu11[1])),10,(0,255,255),4)
                                            #print(round(mu[5],2),"\t",round(self.vtheta,2))
                                            #print(self.theta,"\t",mu2[2])
                                 #cv2.imshow('frame' , self.frame)
##                                 print('a')0
                                 
                                 ch = cv2.waitKey(20)
                                 '''
                                 if len(self.trackers) == 1 :
                                            ch = ord(' ')
                                            print('l')
                                 if len(self.trackers) > 2:
                                            ch = ord(' ')
                                 '''
                                 if ch == ord('w'):
                                     self.closeit = True
                                     print(self.count)
                                     break
                                 if  ch ==  ord(' '):
                                             print('Select Head First then tail')
                                             self.paused = True
                                             if len(self.trackers) > 1:
                                                        self.trackers = []
                                             break
                                 if ch == ord('a'):
                                            print('skipping frames')
                                            self.trackMode = False
                                            break
                                 
##                               time.sleep(0.1)
                                 
                                 ls.remove(ls[0])
                                 ls.remove(ls[0])
                                 
                                 return [[hea1[0],hea1[1],hea2[0],hea2[1]],[ta1[0],ta1[1],ta2[0],ta2[1]]]     
     


class FRAME(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def getframe(self):
       ret,self.frame=cap.read()
       if ret:
           return self.frame
    
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
                                 self.out = True

           def runcode(self):
                      e = True
                      self.count = 0
                      #count=0
                      total = 0
                      initial=False
                      global ls
                      ls1=[[0,0,0,0],[0,0,0,0]]
                      while e :
                                 #print("ls ",ls)
                                 t1 = time.time()
                                 '''
                                 self.lock = threading.Lock()
                                 self.lock.acquire()
                                 ret , self.img = cap.read()
                                 self.lock.release()'''
                                 self.img=self.t.getframe()
                                 #self.img=np.array(self.img).astype(np.float32)
                                 #print(self.img)
                                 
                                 self.count += 1
                                 #print(ret)
                                 #print("Here")
                                 '''
                                 if self.count < 10 and initial:
                                            print(self.count)
                                            time.sleep(1/20)
                                            continue
                                 '''
                                 initial=True
                                 
                                 self.img = cv2.resize(self.img , (640, 380))
                                 print("Det")
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
                                                        #bbox = [int(xyxy[0].item()) , int(xyxy[1].item()) , int(xyxy[2].item() - xyxy[0].item()) , int(xyxy[3].item() - xyxy[1].item()) ]
                                                       if(self.proc==False):
                                                              ls=[]
                                                       #print("Done1")
                                                       ls.append([int(xyxy[0].item()) , int(xyxy[1].item()),int(xyxy[2].item()) , int(xyxy[3].item())])
                                                       #print("Done2")
                                                       self.proc=True
                                                       #print("ls det",len(ls))
                                                       if(len(ls)==2):
                                                           print("Proc Complete")
                                                           ls1=k.kalm(self.img,1)
                                                           if(self.count!=1):
                                                                  self.detFlag=True
                                                           #print(self.detFlag)
                                                           #print("Detect")
                                                           cv2.putText(self.img , "Detecting" , (100 , 140) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                           self.proc=False
                                                       #lis.append(bbox)     
##                                                       bbox = np.array(bbox)
                                                       
                                                 e = False
                                 
                                 #print("ls1",ls1)
                                 
                                 if(len(ls1)==2):
                                        bbox=[int(ls1[0][0]) , int(ls1[0][1]) , int(ls1[0][2] - ls1[0][0]) , int(ls1[0][3] - ls1[0][1])]
                                 
                                 self.trackerhead = cv2.TrackerKCF_create()
                                 #print("Done1")
                                 #print(bbox)
                                 '''
                                 self.lock.acquire()
                                 ret,frame1=cap.read()
                                 self.lock.release()'''
                                 frame1=self.t.getframe()
                                 #print(frame1)
                                 print("Det2")
                                 #print(ret)
                                 frame1=cv2.resize(frame1 , (640, 380))
                                 ok = self.trackerhead.init(frame1 , tuple(bbox))
                                 
                                 #print("Done3")
                                 self.head=True
                                 if(len(ls1)==2):
                                        bbox=[int(ls1[1][0]) , int(ls1[1][1]) , int(ls1[1][2] - ls1[1][0]) , int(ls1[1][3] - ls1[1][1])]
                                 
                                 self.trackertail = cv2.TrackerKCF_create()
                                 
                                 ok = self.trackertail.init(frame1 , tuple(bbox))
                                 self.tail=True
                                 
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
                      #ret,self.frame=cap.read()
                      
                      self.f=FRAME()
                      self.f.start()
                      self.b = DNNThread(opt,self.f)
                      self.b.start()
           '''
           def getframe(self):
                  return self.frame
           '''           
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
                      global ls
                      que1=[]
                      tim=1
                      try :
                                 while True:
                                            print(1)
                                            #print(self.b.detFlag)
                                            if(self.b.detFlag==True):
                                                   print(2)
                                                   while(len(que1)!=0):
                                                          #print("Here Error")
                                                          global ls
                                                          ls=que1[0][0]
                                                          ls1=k.kalm(frame,que1[0][1])
                                                          
                                                          que1.remove(que1[0])
                                                   self.b.detFlag=False
                                                   #print("Done Correct Kalm")
                                                   
                                            #ret , frame = cap.read()
                                            #self.frame=frame
                                            frame=copy.deepcopy(self.f.getframe())
                                            #print(self.frame)
                                            #print("Track1")
                                            print(3)
                                            #print(ret)
                                            frame = cv2.resize(frame , (640, 380))
                                            print("Here")
                                            bgs = bgsub.apply(frame)
                                            print("Here1")
                                            bgs = cv2.erode(bgs,kernel,iterations = 1)
                                            print("Here2")
                                            bgs = cv2.medianBlur(bgs,3)
                                            print("Here3")
                                            bgs = cv2.dilate(bgs,kernel,iterations=2)
                                            print("here4")
                                            bgs = (bgs > 200).astype(np.uint8)*255
                                            print("here5")
                                            
                                            
                                            ts=[]
                                            #if(self.b.head==True):
                                            #          print("IN HEAD")
                                            #self.b.head=False
                                            print(frame)
                                            ok , bbox =  self.b.trackerhead.update(frame)
                                            print("a")
                                            if ok:
                                                       print(4)
                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break
                                                       #print("Head")
                                                       if(self.b.proc==False):
                                                              print(5)
                                                              ts.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 0 , 0 , 255 ) , 2 , 1)
                                            else :
                                                       cv2.putText(frame , "Tracking failure detected in HEAD" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       
                                                       pass
                                                                            
                                            #if(self.b.tail==True):
                                            #           print("IN TAIL")
                                            #           self.b.tail==False
                                            ok , bbox =  self.b.trackertail.update(frame)
                                            print("b")
                                            if ok:
                                                       print(6)
                                                       p1 = ( int ( bbox[0])  , int (bbox[1]) )
                                                       p2 = ( int (bbox[0] + bbox[2]) , int(bbox[1] + bbox[3]))
                                                       if p1[0] < 0 or p1[1] <0 :
                                                                  print('out of bound')
                                                                  print('skipping frames')
                                                                  self.trackMode = False
                                                                  break
                                                       #print("Tail")
                                                       if(self.b.proc==False):
                                                              print(7)
                                                              ts.append([p1[0] , p1[1] , p2[0] , p2[1] ])
                                                       cv2.rectangle(frame , p1 , p2 , ( 255 , 0 , 0 ) , 2 , 1)
                                            else :
                                                       cv2.putText(frame , "Tracking failure detected in TAIL" , (100 , 80) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 255) , 2)
                                                       
                                                       pass
                                            #print("ls tr",len(ls))
                                            #print(ls)
                                            if(len(ts)==2):
                                                print(8)
                                                que1.append((ts,tim))
                                                tim=1
                                                #ls1=k.kalm(frame)
                                                cv2.putText(frame , "Tracking" , (100 , 120) , cv2.FONT_HERSHEY_SIMPLEX , 0.75 , (0 , 0 , 0) , 2)
                                                print("Track")
                                            elif(len(ts)==1):
                                                print(9)
                                                ts.remove(ts[0])
                                                tim+=1
                                            elif(len(ts)==0):
                                                print(10)
                                                tim+=1
                                            #
                                            
                                            
                                                
                                                    
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
                                            cv2.imshow('frame1' , frame)
                                            
                                            self.b.count+=1
                                            #print(self.b.count)
                                            if  cv2.waitKey(20) & 0xff  == 27 or self.b.out :
                                                        self.getout()
                                                        break
                                            #time.sleep(0.53)
                                            time.sleep(1/28)
                                            #print("Track")
                      except Exception as e:
                                             print('Error occured in Tracker thread \n', e )
                                             #self.getout()
                                             self.run()
                                             print(self.is_alive())
                                             pass
                                             print(self.is_alive())
           def getout(self):
                      print('Exiting the program')
                      self.b.stopit = True
                      self.b.join()
                      #cap.stop()
                      out.release()


if __name__ == '__main__':
    k=Kal()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/Python/yolov5/NewYolo/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/Python/yolov5/Video/NewArrow1/NewArrow1.mp4', help='source')  # file/folder, 0 for webcam
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
