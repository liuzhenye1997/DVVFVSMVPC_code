
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import isinsider
import gc
import cv2
import random
import Net
# In[]建立网络

net=Net.Net() 
camera=np.array([1,3,5,7])   
number_of_camera=camera.shape[0]



# In[]训练网络
import projection 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
step=0.00001
optimizer = optim.Adam(net.parameters(), lr=step)
number_sum=0
for epoch in range(2):
    for model in range(18,19): 
        for number_of_photo in range(0,2):         
            for number_of_anim in range(0,4):
              
                for k in range(5):
                    print("C:/Users/liuzhenye/Desktop/Project2/data/SURFACE/model_"+str(model)+"/model_"+str(model)+"_anim_"+str(number_of_anim)+"/00"+str(number_of_photo)+"1.obj")        
                    path = "C:/Users/liuzhenye/Desktop/Project2/data/SURFACE/model_"+str(model)+"/model_"+str(model)+"_anim_"+str(number_of_anim)+"/00"+str(number_of_photo)+"1.obj"
                   
                    bounding_box=projection.bounding_box(path)
                    time1=time.time()
                    #number为一次性训练的点
                    number=800
                    #points为空间中的随机点
                    points=np.zeros([number,3])
                    points[:,0]=np.random.randint(bounding_box[0]-30,bounding_box[1]+30,number)
                    points[:,1]=np.random.randint(bounding_box[2]-30,bounding_box[3]+30,number)
                    points[:,2]=np.random.randint(bounding_box[4]-30,bounding_box[5]+30,number)
#                    points[:,0]=np.random.randint(0,30,number)
#                    points[:,1]=np.random.randint(230,260,number)
#                    points[:,2]=np.random.randint(30,50,number)
                    #labels为空间点相对模型的位置
                    labels=torch.zeros([number,2])        
                    #判断模型是否封闭，若不封闭则跳过
                    openEdges=isinsider.inside(path,number,points,labels)
                    if openEdges!=0:
                        continue  
                    #保证模型内外的点的数量一致
                    value=isinsider.divide(points,labels)
                    points=np.array(value[0])
                    labels=np.array(value[1])
                    #将点顺序打乱
                    index = [i for i in range(points.shape[0])] 
                    random.shuffle(index)
                    points = points[index]
                    labels = labels[index]
            #        print(data,label)
                    
                    print(time.time()-time1)
    
                    number=points.shape[0]
                    number_sum=number_sum+number
                    if number_sum>100000:
                        print(number_sum)
                        number_sum=0
                        step=0.7*step
                        optimizer = optim.Adam(net.parameters(), lr=step)
                    pamt=np.zeros([number_of_camera,3,4])
                    projection_points=np.zeros([number_of_camera,points.shape[0],2])
                    
                    #计算空间点在照片上的投影
                    for i in range(number_of_camera):
                        print(camera[i])
                        pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")       
                        projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],number)
                    
                    photo=torch.empty(number_of_camera,3,1600,1200)
                    for i in range(number_of_camera):                    
                        photo[i,:]=torch.tensor(np.transpose(plt.imread("C:/Users/liuzhenye/Desktop/Project2/data/RENDER/anim_"+str(number_of_anim)+"/model_"+str(model)+"_anim_"+str(number_of_anim)+"/cameras_cam0"+str(camera[i])+"/alpha_00"+str(number_of_photo)+"1.png")).astype(float))
                      
    
                    ticks1 = time.time()
                    running_loss = 0.0
                    photo=photo.to(device)
    
                   
                      
                    inputs=torch.tensor(projection_points)
    
                    inputs = inputs.to(device)
                    label=torch.tensor(labels).cuda().type(torch.float)
                    lable=label.to(device)
        
                    net=net.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    time1=time.time()
                    outputs = net(photo,inputs).type(torch.float)
                    print(time.time()-time1)
                    loss=torch.nn.functional.binary_cross_entropy(outputs,label)
                    print(loss)
    
                    loss.backward()
                    optimizer.step()
     
                    running_loss += loss.item()
                    print('loss:',loss)   
                    np.savetxt(str(number_of_camera)+'_'+str(epoch)+'_'+ str(model)+'_'+str(number_of_photo)+'.txt',np.array([running_loss/number]))
                        
    
                    running_loss = 0.0
                    print(time.time()-ticks1)
                    #保存训练结果
                    torch.save({
                    'epoch': epoch,
                    'model':model,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,            
                    },"../test15/"+str(number_of_camera)+'_'+str(epoch)+'_'+ str(model)+'_'+str(number_of_photo)+'_'+str(number_of_anim)+'.pt')
                    torch.cuda.empty_cache()
                    gc.collect()
print('Finished Training')



# In[]用于加载训练了一半的模型

optimizer = (optim.SGD(net.parameters(), lr=0.0001, momentum=0.9))
checkpoint = torch.load('4_0_17_0_2.pt')
net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_temp = checkpoint['epoch']
loss = checkpoint['loss']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=net.to(device)
net.train()
#net.eval()
print(epoch_temp)    
    
    
# In[]使用通过模型得到的概率预测空间点相对模型的位置
import numpy as np
import matplotlib.pyplot as plt
def Position(probability):
    a=probability[0]*(1-probability[1])
    b=probability[0]*probability[1]
    c=probability[1]*(1-probability[0])
    if b>=a and b>=c:
        return [1,1]
    elif c>=a and c>=b:
        return [0,1]
    else:    
        return [1,0]
 

 


# In[]测试训练得到的模型，加载测试集
import numpy as np
import projection
path = "./data_test/model_19_anim_0/0001.obj"
model=2
bounding_box=projection.bounding_box(path)
print(bounding_box)
time1=time.time()

number=500
points=np.zeros([number,3])
points[:,0]=np.random.randint(bounding_box[0],bounding_box[1],number)
points[:,1]=np.random.randint(bounding_box[2],bounding_box[3],number)
points[:,2]=np.random.randint(bounding_box[4],bounding_box[5],number)
labels=torch.zeros([number,2])            
openEdges=isinsider.inside(path,number,points,labels)
             
value=isinsider.divide(points,labels)
points=np.array(value[0])
labels=np.array(value[1])
index = [i for i in range(points.shape[0])] 
random.shuffle(index)
points = points[index]
labels = labels[index]

#number=20
#print(number)
#points=np.random.randint(100,101,(number,3))
##points=numpy.array([[-25,260,110]])
##print(points)
#
#for i in range(number):
#    points[i,:]=np.array([5,245+i,80])
#labels=torch.zeros([number,2])
labels=np.zeros([number,2])
path = "C:/Users/liuzhenye/Desktop/Project2/data/SURFACE/model_"+str(19)+"/model_"+str(19)+"_anim_"+str(0)+"/00"+str(0)+"1.obj"
number=points.shape[0]
print(projection.bounding_box(path))
isinsider.inside(path,number,points,labels) 
print(labels)

print(time.time()-time1)

#            points=np.load("points_"+str(model)+'_deleted_'+str(number_of_photo)+'.npy')
##        #    print(points)
##           
#            labels=np.load("labels_"+str(model)+'_deleted_'+str(number_of_photo)+'.npy')
number=points.shape[0]

pamt=np.zeros([number_of_camera,3,4])
projection_points=np.zeros([number_of_camera,number,2])
for i in range(number_of_camera):
    pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")       
    projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],number)
photo=torch.empty(number_of_camera,3,1600,1200)        
for i in range(number_of_camera):    
    photo[i,:]=torch.tensor(np.transpose(plt.imread("./data_test/model_19_anim_0/cameras_cam0"+str(camera[i])+"/alpha_0001.png")).astype(float))

# In[]测试训练得到的模型，计算预测空间点位置的正确率
print(labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=net.to(device)
correct=0
with torch.no_grad():
    

    inputs=torch.tensor(projection_points)
#        inputs=torch.rand(-100,100,[number,3])
    inputs=inputs.to(device)
   
    photo=photo.to(device)
    outputs = net(photo,inputs)
#    loss=torch.nn.functional.binary_cross_entropy(outputs,labels)
        
    for i in range(number):
        a=torch.zeros([1,2])
        a=Position([outputs[i,0],outputs[i,1]])
        if a[0]==labels[i,0] and a[1]==labels[i,1]:
            correct=correct+1
#            print(111111111111)
#        print(outputs)
    
        else:
             print(outputs[i,:],labels[i,:])
             print(loss)
        if labels[i,0]==1 and labels[i,1]==1:
            print(111111)
print(correct,correct/number)



# In[]清空显存
gc.collect()
del photo
del net
del projection_points
del pamt
del inputs

gc.collect()
torch.cuda.empty_cache()  



# In[]后面的是写重建模型部分遗留下的草稿，实际上用不上 
import isinsider
import projection
number=40
path = "./data_test/model_19_anim_0/0001.obj"
bounding_box=projection.bounding_box(path)
print(bounding_box)


 
photo=torch.empty(number_of_camera,3,1600,1200)        
for i in range(number_of_camera):    
    photo[i,:]=torch.tensor(np.transpose(plt.imread("./data_test/model_19_anim_0/cameras_cam0"+str(camera[i])+"/alpha_0001.png")).astype(float))
photo=photo.to(device)
bounding_box=projection.bounding_box(path)
points=np.zeros([number*number*number,3])
labels=np.zeros([number*number*number,2])
for i in range(number):
    for j in range(number):
        for k in range(number):
            points[number*number*i+number*j+k,:]=[(bounding_box[1]-bounding_box[0])*i/(number-1)+bounding_box[0],(bounding_box[3]-bounding_box[2])*j/(number-1)+bounding_box[2],(bounding_box[5]-bounding_box[4])*k/(number-1)+bounding_box[4]]
np.savetxt('all_points.txt',points)             
          
projection_points=np.zeros([number_of_camera,number*number*number,2])
pamt=np.zeros([number_of_camera,3,4])
for i in range(number_of_camera):
    pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")       
    projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],number*number*number)
print(1111)                   
with torch.no_grad():
    inputs=torch.tensor(projection_points).cuda()
    inputs=inputs.to(device)  
    time1=time.time()
    outputs = net(photo,inputs)
    
#    outputs=isinsider.inside(path,number*number*number,points,labels)
# In[] 
single_label= np.zeros([number,number,number])   
print(number)
for i in range(number):
    for j in range(number):
        for k in range(number):      
#                print(points[i*number*number+j*number+k,:])
           
         
        
            a=torch.zeros([1,2])
            a=Position([outputs[i*number*number+j*number+k,0],outputs[i*number*number+j*number+k,1]])
                           
            labels[i*number*number+j*number+k,:]=a
            if labels[i*number*number+j*number+k,0]==1 and labels[i*number*number+j*number+k,1]==1:
                single_label[i,j,k]=0
            elif labels[i*number*number+j*number+k,0]==1 and labels[i*number*number+j*number+k,1]==0:    
                single_label[i,j,k]=-1
            else:
                single_label[i,j,k]=1
np.save('leabels',single_label)      
# In[]
               
points_surface=[]
points_inside=[]
points_outside=[]
points_surface_one=[]
points_surface_two=[]
for i in range(number):
    for j in range(number):
        for k in range(number):
            if labels[i*number*number+j*number+k,0]==1 and labels[i*number*number+j*number+k,1]==1:
                points_surface.append(points[i*number*number+j*number+k,:])
                points_surface_one.append(points[i*number*number+j*number+k,:])
for i in range(number-1):
    for j in range(number-1):
        for k in range(number-1):   
            sum_outside=0
            sum_inside=0
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
#                        print(labels[(i+ii)*number*number+(j+jj)*number+k+kk,:])
                        sum_inside=sum_inside+labels[(i+ii)*number*number+(j+jj)*number+k+kk,0]
                        sum_outside=sum_outside+labels[(i+ii)*number*number+(j+jj)*number+k+kk,1]
#            print(sum_outside,sum_inside)            
            if sum_outside==0 and sum_inside==8:
                continue
            elif sum_inside==0 and sum_outside==8:
                continue
            elif sum_inside+sum_outside==8:
                point=(points[(i)*number*number+(j)*number+k,:]+points[(i+1)*number*number+(j+1)*number+k+1,:])/2
                points_surface.append(point)  
                points_surface_two.append(point)
# In[]

a=np.zeros([len(points_surface),3])
for i in range(len(points_surface)):
    a[i,:]=points_surface[i]
#    print(points_surface[i])
print(a)    
np.savetxt('./point/points_surface_1357.txt',a)    
np.savetxt('./point/points_surface_one_1357.txt',points_surface_one)
np.savetxt('./point/points_surface_two_1357.txt',points_surface_two)  
# In[]
print(labels)


