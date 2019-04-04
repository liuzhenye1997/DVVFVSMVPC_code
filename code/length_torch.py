# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:00:55 2018

@author: liuzhenye
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:32:32 2018

@author: liuzhenye
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:21:20 2018

@author: liuzhenye
"""
import numpy as np
import math
import time
import torch
import numba as nb
#计算空间点到模型的距离，为了使用numba库把函数拆开了，显得支离破碎。与length.py的区别是混合使用了torch和numpy。
#根据我测试一般情况下使用torch的运行速度比使用numpy快。可能是对并行计算优化更好。

#@nb.vectorize(nopython=True)
#@nb.jit(nopython=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def crossing_s(a,b):
    c=torch.empty((a.shape[0],3))
    c[:,0]=a[:,1]*b[:,2]-a[:,2]*b[:,1]
    c[:,1]=a[:,2]*b[:,0]-a[:,0]*b[:,2]
    c[:,2]=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    return c

@nb.jit(nopython=True)
def crossing(a,b):  
    c=np.zeros(3)
    c[0]=a[1]*b[2]-a[2]*b[1]
    c[1]=a[2]*b[0]-a[0]*b[2]
    c[2]=a[0]*b[1]-a[1]*b[0]
    return c
#@nb.jit(nopython=True)
#@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],'(n),(n)->(n)')
#def sub(a,b,c):  
#    for i in range(3):
#        c[i]=a[i]-b[i]
      

#@nb.jit(nopython=True)
#def equal(a,b):
##    print(a,b)
#    if a[0]*b[0]+a[1]*b[1]+a[2]*b[2]<0:
#        
#        return False
#    else:
##        print(1111)
#        return True
    
#    a=a/np.sqrt(dot(a[0],0,a[1],0,a[2],0))
#    b=b/np.sqrt(dot(b[0],0,b[1],0,b[2],0))
#    for i in range(3):
#        if math.fabs(a[i]-b[i])>0.00000001:
#            return False
#    return True  
  
@nb.jit(nopython=True)
def length_to_edge(a,b,point):
#    print(a,b,point)
#    if np.dot(b-a,point-a)<0:
#        return dot(point[0],a[0],point[1],a[1],point[2],a[2])
#    elif np.dot(a-b,point-b)<0:
##        return np.sqrt(dot(point[0],b[0],point[1],b[1],point[2],b[2]))
#        return dot(point[0],b[0],point[1],b[1],point[2],b[2])
    c=crossing(point-a,point-b)
    d=b-a
    return dot(c[0],0,c[1],0,c[2],0)/dot(d[0],0,d[1],0,d[2],0)


def length(number,triangles,point):
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    triangles.to(device)
#    
#    point.to(device)
#    time1=time.time()
    point=point.type(torch.float)
    triangles=triangles.type(torch.float)
    a=triangles[:,1,:]-triangles[:,0,:]
    b=triangles[:,2,:]-triangles[:,0,:]
    c=crossing_s(a,b)
#    print(time.time()-time1)
    
    m=torch.zeros(number).type(torch.float)   
    for i in range(3):
       m=m+triangles[:,0,i]*c[:,i]
    length1=(c[:,0]*point[0]+c[:,1]*point[1] +c[:,2]*point[2]-m)/torch.sqrt(c[:,0]*c[:,0]+c[:,1]*c[:,1]+c[:,2]*c[:,2]).type(torch.float)  
    k=(c[:,0]*point[0]+c[:,1]*point[1] +c[:,2]*point[2]-m)/(c[:,0]*c[:,0]+c[:,1]*c[:,1]+c[:,2]*c[:,2])  
   
#    print(time.time()-time1)
    
    d=torch.empty((triangles.shape[0],3))
    for i in range(3):
        d[:,i]=triangles[:,2,i]-triangles[:,1,i]    
    for i in range(3):
        for j in range(3):
            triangles[:,i,j]=triangles[:,i,j]+k*c[:,j]  
    
#    print(time.time()-time1)
        
    vector=point-triangles
    cross=torch.empty((number,3,3))    
    cross[:,0,:]=crossing_s(a[:,:],vector[:,0,:])
    cross[:,1,:]=crossing_s(d[:,:],vector[:,1,:])
    cross[:,2,:]=crossing_s(-b[:,:],vector[:,2,:])
    
#    print(time.time()-time1)
    
    cross_dot=torch.empty((number,3))  
    for i in range(3):
        cross_dot[:,i]=-(cross[:,i,0]*c[:,0]+cross[:,i,1]*c[:,1]+cross[:,i,2]*c[:,2])
    distance=torch.empty((number,3))  
    for i in range(3):
        distance[:,i]=torch.norm(point-triangles[:,i,:],2,1)
   
    distance_max=torch.empty(number)
    distance_max=torch.max(distance,1)[0]    
    
    vector=torch.empty((number,3,3))
    for i in range(3):
        for j in range(3):
            vector[:,i,j]=(triangles[:,i,0]-triangles[:,j,0])*(point[0]-triangles[:,j,0])+(triangles[:,i,1]-triangles[:,j,1])*(point[1]-triangles[:,j,1])+(triangles[:,i,2]-triangles[:,j,2])*(point[2]-triangles[:,j,2])
    
#    print(time.time()-time1)
#    time1=time.time()               
    point=point.numpy()
    triangles=(triangles.numpy()).astype(float)
#    a=(a.numpy()).astype(float)
#    b=(b.numpy()).astype(float)
#    c=(c.numpy()).astype(float)
#    d=(d.numpy()).astype(float)   
    distance_max=(distance_max).numpy().astype(float)  
    vector=(vector).numpy().astype(float)  
    length1=length1.numpy().astype(float)  
    cross_dot=cross_dot.numpy().astype(float)  
#    print(time.time()-time1)      
    time1=time.time()
    min_=length_min(cross_dot,length1,number,point,triangles,distance_max,vector)
#    print(time.time()-time1)        
    return min_
 
@nb.jit(nopython=True)
def length_min(cross_dot,length1,number,point,triangles,distance_max,vector):
     min_=10000.0 
#     np.dot(b-a,point-a)<0:
#        return dot(point[0],a[0],point[1],a[1],point[2],a[2])
#    elif np.dot(a-b,point-b)<0:
     for i in range(number): 
        if cross_dot[i,0]>0 and vector[i,1,0]>0 and vector[i,0,1]>0:            
            length2=length_to_edge(triangles[i,0,:],triangles[i,1,:],point)
        elif cross_dot[i,1]>0  and vector[i,2,1] and vector[i,1,1]>0:
            length2=length_to_edge(triangles[i,1,:],triangles[i,2,:],point)
        elif cross_dot[i,2]>0 and vector[i,2,0]>0 and vector[i,0,2]>0:
            length2=length_to_edge(triangles[i,0,:],triangles[i,2,:],point)
        elif cross_dot[i,0]>0 or cross_dot[i,1]>0 or  cross_dot[i,2]>0:
            length2=distance_max[i]*distance_max[i]
        else:
            length2=0 
#        print(length1,length2)    
#        print(time.time()-time1)    
        
        length=np.sqrt(length1[i]*length1[i]+length2)
        if length <min_:
            min_=length
      
     return min_        
    


#    
#import time
#
##time1=time.time()
#number_of_triangle=1
##for i in range(100):
#
##triangle=np.random.rand(number_of_triangle,3,3)
#triangle=np.array([[[-1,-1,0],[1,-1,0],[0,1,0]]]).astype(float)
#point=np.array([-2,-2,-1])
#
#
##    print(length(triangle,point)) 
#
#print(length(number_of_triangle,triangle,point))

#    if length(number_of_triangle,triangle,point)<19 or length(number_of_triangle,triangle,point)>20:
#        break
#    print(triangle.dtype,point.dtype)   
#print(time.time()-time1) 
#print(time1)      
#       



#a=np.zeros(3)
#b=np.ones(3)
#c=np.zeros(3)
#sub(a,b,c)
#print(c)













#@nb.jit(nopython=True)
def length_to_point(number,triangle,point):
    min=1000000
    for i in range(number):
#        min_temp=np.dot(point-triangle[i,:],point-triangle[i,:])
#        print(np.dot(point-triangle[i,:],point-triangle[i,:]))
        min_temp=dot(point[0],triangle[i,0],point[1],triangle[i,1],point[2],triangle[i,2])
        if(min>min_temp):
            min=min_temp
    return min



@nb.vectorize(nopython=True)
def dot(a,b,c,d,e,f):
    return (a-b)*(a-b)+(c-d)*(c-d)+(e-f)*(e-f)


