# -*- coding: utf-8 -*-


import isinsider
import projection
import numpy as np
import mcubes
import time
import numba as nb
from Net import Net
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

#使用通过模型得到的概率预测空间点相对模型的位置，与init.py里的同名函数是同一个函数
@nb.jit(nopython=True)
def Position(probability):
    a=probability[0]*(1-probability[1])
    b=probability[0]*probability[1]
    c=probability[1]*(1-probability[0])
    if b>=a and b>=c:
        return np.array([1,1])
    elif c>=a and c>=b:
        return np.array([0,1])
    else:    
        return np.array([1,0])

#marching cube算法，使用的是从网上找的mcubes库提供的
def marching_cube(resolution,bounding_box,number_fo_anim):
    u= np.load('leabels.npy')
    print(u.shape)
    
      # Extract the 0-isosurface
    vertices, triangles = mcubes.marching_cubes(u, 0.1)
    print(np.max(vertices[:,2])) 

    print(bounding_box,resolution)
    for i in range(vertices.shape[0]):
        for j in range(3):
            vertices[i,j]=vertices[i,j]/resolution*(bounding_box[2*j+1]-bounding_box[2*j])+bounding_box[2*j]
    print(np.max(vertices[:,1]))        
    mcubes.export_mesh(vertices, triangles, "sphere_"+str(number_fo_anim)+".dae", "MySphere")
    return vertices




 
#def accurate_model(bounding_box,number,path):
#    points=np.zeros([number*number*number,3])
#    labels=np.zeros([number*number*number,2])
##    time1=time.time()
#   set_points(bounding_box,points)          
##    print(time.time()-time1)          
#
#                 
##    print(1111)
##    time1=time.time()    
#    outputs=isinsider.inside(path,number*number*number,points,labels)
##    print(time.time()-time1)
#    return labels

#在bounding box空间中建立均匀的三维点阵
@nb.jit(nopython=True) 
def set_points(bounding_box,number,points):
#    print(bounding_box)
    for i in range(number):
        for j in range(number):
            for k in range(number):
                points[number*number*i+number*j+k,:]=np.array([(bounding_box[1]-bounding_box[0])*i/(number-1)+bounding_box[0],(bounding_box[3]-bounding_box[2])*j/(number-1)+bounding_box[2],(bounding_box[5]-bounding_box[4])*k/(number-1)+bounding_box[4]])
    return points

#将空间中的点送入网络计算概率
def prediction_model(number_of_camera,number,net,bounding_box):
#    time1=time.time()
    points=np.zeros([number*number*number,3])
    set_points(bounding_box,number,points)
    projection_points=np.zeros([number_of_camera,number*number*number,2])
    pamt=np.zeros([number_of_camera,3,4])
    
    for i in range(number_of_camera):
        pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")       
        projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],number*number*number)    
                  
    with torch.no_grad():
        inputs=torch.tensor(projection_points)
        inputs=inputs.to(device)  
#        print(time.time()-time1)       
        outputs = net(photo,inputs)
    return (outputs,points)   

#使用通过模型得到的概率预测空间点相对模型的位置，与上面的同样功能函数的区别在于一个使用二维参数表示相对位置，一个使用一个参数。
#现在再来看没有必要使用两个函数，直接合并成同一个就好。    
@nb.jit(nopython=True)
def  prediction_model_label(number,outputs,single_label,labels):
    for i in range(number):
        for j in range(number):
            for k in range(number):                            
            
                a=Position(outputs[i*number*number+j*number+k,:])
                labels[i*number*number+j*number+k,:]=a               
            
                if a[0]==1 and a[1]==1:
                    single_label[i,j,k]=0
                elif a[0]==1 and a[1]==0:    
                    single_label[i,j,k]=-1
                else:
                    single_label[i,j,k]=1
                    
#给single_label全部赋值label_number。不过现在来看好像一行就解决了，为何还要这个函数。
@nb.jit(nopython=True)                    
def prediction_model_label_same(number,single_label,label_number):
    for i in range(number):
        for j in range(number):
            for k in range(number):                                           
                single_label[i,j,k]=label_number
              
                 


# In[]使用训练得到的网络将表面，内部和外部点计算出来。事实上，如果我理解没错的话，文章先是将bounding_box细分一次，然后对需要继续细分的空间不断细分。
#但是当我写代码的时候，感觉难度太高，无法保证在短时间内在多次细分。于是只细分了两处。这也许是我生成的模型非常粗糙的原因之一。               

model=19
#number_fo_anim=3
number_of_photo=0
number_of_camera=4
camera=np.array([1,3,5,7])   
for number_fo_anim in range(1):
    path = "C:/Users/liuzhenye/Desktop/Project2/data/SURFACE/model_"+str(model)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/00"+str(number_of_photo)+"1.obj"
    bounding_box=np.array(projection.bounding_box(path))
    for i in range(3):
        bounding_box[2*i]=bounding_box[2*i]-10
        bounding_box[2*i+1]=bounding_box[2*i+1]+10
    print(bounding_box)    
    
    #def prediction_model(bounding_box,number,photo):
    net=Net()
    optimizer = (optim.SGD(net.parameters(), lr=0.001, momentum=0.9))
    checkpoint = torch.load('4_0_17_0_2.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_temp = checkpoint['epoch']
    loss = checkpoint['loss']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=net.to(device)
    
    print(epoch_temp) 
    
    number1=32
    number2=4
    number=number1*number2
    
   
    print(path) 
    photo=torch.empty(number_of_camera,3,1600,1200)        
    for i in range(number_of_camera):    
        photo[i,:]=torch.tensor(np.transpose(plt.imread("C:/Users/liuzhenye/Desktop/Project2/data/RENDER/anim_"+str(number_fo_anim)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/cameras_cam0"+str(camera[i])+"/alpha_00"+str(number_of_photo)+"1.png")).astype(float)) 
        print("C:/Users/liuzhenye/Desktop/Project2/data/RENDER/anim_"+str(number_fo_anim)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/cameras_cam0"+str(camera[i])+"/alpha_00"+str(number_of_photo)+"1.png") 
    photo=photo.to(device)
    
               
              
    
        
    
     
    
    time2=time.time()
    single_label_mesh= np.zeros([number1,number1,number1])   
    single_label_center= np.zeros([number1-1,number1-1,number1-1])  
    labels_mesh= np.zeros([number1*number1*number1,2])   
    labels_center= np.zeros([number1*number1*number1,2])   
    single_label=np.ones([number,number,number]) 
       
    
    (outputs_center,points)=prediction_model(number_of_camera,number1-1,net,bounding_box/number1*(number-1)) 
    outputs_center=np.array(outputs_center)
    time1=time.time()
    (outputs_mesh,grid_points)=prediction_model(number_of_camera,number1,net,bounding_box) 
    print(time.time()-time1)
    outputs_mesh=np.array(outputs_mesh)
    
    prediction_model_label(number1,outputs_mesh,single_label_mesh,labels_mesh)
    prediction_model_label(number1-1,outputs_center,single_label_center,labels_center)
    print(grid_points.shape)
    
    boundary_list=[]
    range_list=[]
    print('111111111111111111111111',time.time()-time2)
    time2=time.time()
    times=0
    for i in range(number1-1):
        for j in range(number1-1):
            for k in range(number1-1):
                sum_inside=0
                sum_outside=0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            sum_inside=sum_inside+labels_mesh[(i+ii)*number1*number1+(j+jj)*number1+k+kk,0]
                            sum_outside=sum_outside+labels_mesh[(i+ii)*number1*number1+(j+jj)*number1+k+kk,1]
    #            print(sum_outside,sum_inside) 
                if  single_label_center[i,j,k]==0: 
                    times=times+1
                    boundary=np.zeros(6)
    #                print(grid_points.shape)
                    for m in range(3):                    
                        boundary[2*m]=grid_points[(i)*number1*number1+(j)*number1+k,m]
                        boundary[2*m+1]=grid_points[(i+1)*number1*number1+(j+1)*number1+k+1,m]
                    boundary_list.append(boundary)    
                    range_list.append([i,j,k])
                    
                    
                   
                    continue
                elif sum_outside==0 and sum_inside==8:
                    prediction_model_label_same(number2,single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],-1)
                    continue
                elif sum_inside==0 and sum_outside==8:
                    prediction_model_label_same(number2,single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],1)
                    continue
                else :
                    times=times+1
                    boundary=np.zeros(6)
                    for m in range(3):                    
                        boundary[2*m]=grid_points[(i)*number1*number1+(j)*number1+k,m]
                        boundary[2*m+1]=grid_points[(i+1)*number1*number1+(j+1)*number1+k+1,m]
                    boundary_list.append(boundary)      
                    range_list.append([i,j,k])
                                           
                    
                    continue
    print('111111111111111111111111',time.time()-time2)   
    time2=time.time()                 
    batch_size=4*4*4
    number_of_batch=int((len(boundary_list)-1)/batch_size)+1
    print(number_of_batch,number1,number2)
    for batch in range(number_of_batch):
        if batch!=number_of_batch-1:
            size=batch_size
        else:
            size=len(boundary_list)-batch_size*(number_of_batch-1)
    #    time1=time.time()
        points=np.zeros([number2*number2*number2*size,3])
        for i in range(size):
            set_points(boundary_list[batch*batch_size+i],number2,points[i*number2*number2*number2:(i+1)*number2*number2*number2,:])
            
        projection_points=np.zeros([number_of_camera,number2*number2*number2*size,2])
        pamt=np.zeros([number_of_camera,3,4])
    
        for i in range(number_of_camera):
            pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")       
            projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],number2*number2*number2*size)    
                  
        with torch.no_grad():
            inputs=torch.tensor(projection_points)
            inputs=inputs.to(device)        
            output = net(photo,inputs)
        output=np.array(output)
    #        output=np.zeros([number2*number2*number2*size,2])
    #    print(output.shape)
        labels_temp= np.zeros([number2*number2*number2,2])
        for m in range(size):
            i=range_list[batch*batch_size+m][0]
            j=range_list[batch*batch_size+m][1]
            k=range_list[batch*batch_size+m][2]
    
            prediction_model_label(number2,output[m*number2*number2*number2:(m+1)*number2*number2*number2,:],single_label[number2*i:number2*(i+1),number2*j:number2*(j+1),number2*k:number2*(k+1)],labels_temp)                  
    print('111111111111111111111111',time.time()-time2,times)
                 
    np.save('leabels',single_label)   
        
                        
    time1=time.time()
    
    print(time.time()-time1)     
    print(33333333333333333)           
    #                print(111)               
# In[]对表面点使用marching cube算法进行三维重建
  
vertices=marching_cube(number,bounding_box,number_fo_anim) 
