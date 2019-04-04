# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:15:20 2018

@author: liuzhenye
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
#网络
n=1
m=3
number_of_camera=4
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(m,m),padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(8, 8, kernel_size=(m, m),padding=1)
        self.conv2_2 = nn.Conv2d(8, 16, kernel_size=(m, m),padding=1)
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=(m, m),padding=1)
        self.conv3_2 = nn.Conv2d(16, 32, kernel_size=(m, m),padding=1)
        self.conv4_1 = nn.Conv2d(32, 32, kernel_size=(m, m),padding=1)
        self.conv4_2 = nn.Conv2d(32, 64, kernel_size=(m, m),padding=1)
        self.conv5_1 = nn.Conv2d(64, 64, kernel_size=(m, m),padding=1)
        self.conv5_2 = nn.Conv2d(64, 128, kernel_size=(m, m),padding=1)
        self.conv6_1 = nn.Conv2d(128,128, kernel_size=(m, m),padding=1)
        self.conv6_2 = nn.Conv2d(128, 256, kernel_size=(m, m),padding=1)
        self.conv7_1 = nn.Conv2d(256,256, kernel_size=(m, m),padding=1)
        self.conv7_2 = nn.Conv2d(256, 512, kernel_size=(m, m),padding=1)
        self.fc1 =nn.Linear(1016, 512)
        self.fc2 =nn.Linear(512, 512)
        self.fc3 =nn.Linear(512, 1024)
        self.pool2 =nn.MaxPool2d((number_of_camera,1), 1)
        self.pool3 =nn.AvgPool2d((number_of_camera,1), 1)
        self.fc4 =nn.Linear(2048, 512)
        self.fc5 =nn.Linear(512, 128)
        self.fc6 =nn.Linear(128,2)
        
        self.bn1=nn.BatchNorm2d(8,affine=True)
        self.bn2_1=nn.BatchNorm2d(8,affine=True)
        self.bn2_2=nn.BatchNorm2d(16,affine=True)
        self.bn3_1=nn.BatchNorm2d(16,affine=True)
        self.bn3_2=nn.BatchNorm2d(32,affine=True)
        self.bn4_1=nn.BatchNorm2d(32,affine=True)
        self.bn4_2=nn.BatchNorm2d(64,affine=True)
        self.bn5_1=nn.BatchNorm2d(64,affine=True)
        self.bn5_2=nn.BatchNorm2d(128,affine=True)
        self.bn6_1=nn.BatchNorm2d(128,affine=True)
        self.bn6_2=nn.BatchNorm2d(256,affine=True)
        self.bn7_1=nn.BatchNorm2d(256,affine=True)
        self.bn7_2=nn.BatchNorm2d(512,affine=True)
        
    #photo为输入相片，projection_points为空间点在照片上的投影    
    def forward(self, photo,projection_points):
        # per-view feature extraction
        time1=time.time()
        number_of_points=projection_points.shape[1]

        x_ceil=torch.zeros(number_of_points).cuda()
        x_floor=torch.zeros(number_of_points).cuda()
        y_ceil=torch.zeros(number_of_points).cuda()
        y_floor=torch.zeros(number_of_points).cuda()
     

        x=photo[:,:,:,:]
     
        projection_x=projection_points[:,:,0].type(torch.float)
        projection_y=projection_points[:,:,1].type(torch.float)
        y=torch.zeros([number_of_camera,number_of_points,1016]).cuda()
        
        x = self.conv1(x)            
        x = self.bn1(x)
        
       
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        #is_outside用于判别projection_points是否在照片内部
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)

        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
                
        for i in range(number_of_camera):
            y[i,:,0:8]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                    +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                    +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                    +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
            
            
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)           
        x = F.relu(self.bn2_1(self.conv2_1(x)))            
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,8:24]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                    +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                    +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                    +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
        
        
#            print(3)
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)        
        x = F.relu(self.bn3_1(self.conv3_1(x)))            
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,24:56]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
        
        
#            print(4)
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,56:120]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
        
            
#            print(5)
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))      
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,120:248]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                        +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                        +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
  
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)
        x = F.relu(self.bn6_1(self.conv6_1(x)))         
        x = F.relu(self.bn6_2(self.conv6_2(x)))             
        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,248:504]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                            +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                            +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                            +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
  
        
     
        projection_x=projection_x/2
        projection_y=projection_y/2
        x = self.pool(x)
        x = F.relu(self.bn7_1(self.conv7_1(x)))            
        x = F.relu(self.bn7_2(self.conv7_2(x)))  

        x_ceil=torch.ceil(projection_x)
        x_floor=x_ceil-1
        y_ceil=torch.ceil(projection_y)
        y_floor=y_ceil-1 
        is_outside=((projection_x<=(x.shape[2]-1))*(projection_x>=0)*(projection_y<=(x.shape[3]-1))*(projection_y>=0)).type(torch.float)
        x_ceil=is_outside*x_ceil
        x_floor=is_outside*x_floor
        y_ceil=is_outside*y_ceil
        y_floor=is_outside*y_floor
        for i in range(number_of_camera):
            y[i,:,504:1016]=((x[i,:,x_ceil[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(y_floor[i,:]-projection_y[i,:])
                            +x[i,:,x_floor[i,:].type(torch.long),y_ceil[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(y_floor[i,:]-projection_y[i,:])
                            +x[i,:,x_ceil[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(x_floor[i,:]-projection_x[i,:])*(projection_y[i,:]-y_ceil[i,:])
                            +x[i,:,x_floor[i,:].type(torch.long),y_floor[i,:].type(torch.long)]*(projection_x[i,:]-x_ceil[i,:])*(projection_y[i,:]-y_ceil[i,:])).transpose(0,1))    
  
        #multi-view feature fusion and classification

        print(x.shape)
        x=y
        print(x.shape) 
        x = F.relu(self.fc1(x))
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)  
        x=self.fc3(x)

   

        x=x.transpose(1,0)

        x1 = self.pool2(x)
        x2 = self.pool3(x)
        x=torch.cat((x1,x2),2)
        x=x.view(-1,2048)
#        print(x.shape)
        x = F.relu(self.fc4(x))
        x=F.dropout(F.relu(self.fc5(x)), p=0.5, training=self.training)
        x = torch.sigmoid(self.fc6(x))

        return x
        

#为了并行化留下的一个函数，现在并没有什么用
def cycle(number_of_points, x_ceil,x_floor,y_ceil,y_floor,projection_x,projection_y,shape_3,shape_2):
    time1=time.time()
    for j in range(number_of_points):               
        if  projection_x[j]>shape_2-1 or projection_x[j]<0 or projection_y[j]>shape_3-1 and projection_y[j]<0:                                                      
            x_ceil[j]=x_floor[j]=y_ceil[j]=y_floor[j]=0
    print(time.time()-time1)