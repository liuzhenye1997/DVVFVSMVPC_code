# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:21:20 2018

@author: liuzhenye
"""
import numpy as np
import math
import time
import numba as nb
#计算空间点到模型的距离，为了使用numba库把函数拆开了，显得支离破碎

#@nb.vectorize(nopython=True)
@nb.jit(nopython=True)
def crossing(a,b):
    c=np.zeros(3)
    c[0]=a[1]*b[2]-a[2]*b[1]
    c[1]=a[2]*b[0]-a[0]*b[2]
    c[2]=a[0]*b[1]-a[1]*b[0]
    return c

#@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],'(n),(n)->(n)')
#def crossing_s(a,b,c):  
#    c[0]=a[1]*b[2]-a[2]*b[1]
#    c[1]=a[2]*b[0]-a[0]*b[2]
#    c[2]=a[0]*b[1]-a[1]*b[0]
#@nb.jit(nopython=True)
#@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],'(n),(n)->(n)')
#def sub(a,b,c):  
#    for i in range(3):
#        c[i]=a[i]-b[i]
      

@nb.jit(nopython=True)
def equal(a,b):
#    print(a,b)
    if a[0]*b[0]+a[1]*b[1]+a[2]*b[2]<0:
        
        return False
    else:
#        print(1111)
        return True
    
#    a=a/np.sqrt(dot(a[0],0,a[1],0,a[2],0))
#    b=b/np.sqrt(dot(b[0],0,b[1],0,b[2],0))
#    for i in range(3):
#        if math.fabs(a[i]-b[i])>0.00000001:
#            return False
#    return True  
  
@nb.jit(nopython=True)
def length_to_edge(a,b,point):
#    print(a,b,point)
    if np.dot(b-a,point-a)<0:
        return dot(point[0],a[0],point[1],a[1],point[2],a[2])
    elif np.dot(a-b,point-b)<0:
#        return np.sqrt(dot(point[0],b[0],point[1],b[1],point[2],b[2]))
        return dot(point[0],b[0],point[1],b[1],point[2],b[2])
    c=crossing(point-a,point-b)
    d=b-a
    return dot(c[0],0,c[1],0,c[2],0)/dot(d[0],0,d[1],0,d[2],0)

@nb.jit(nopython=True)
def length(number,triangles,point):
    min=10000  
    for i in range(number):
        triangle=triangles[i,:,:]
        min_temp=length_to_triangle(triangle,point)
        if min_temp<min:
            min=min_temp
    return min        
 


@nb.jit(nopython=True)
def length_to_triangle(triangle,point):
#    time1=time.time()
    a=triangle[1,:]-triangle[0,:]
    b=triangle[2,:]-triangle[0,:]
    c=crossing(a,b)
    m=0
#    print(time.time()-time1)   
    for i in range(3):
       m=m+triangle[0,i]*c[i]
    length1=(c[0]*point[0]+c[1]*point[1] +c[2]*point[2]-m)/math.sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])  
    k=(c[0]*point[0]+c[1]*point[1] +c[2]*point[2]-m)/(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])  
#    print(time.time()-time1)   
    d=np.zeros(3)
    for i in range(3):
        d[i]=triangle[2,i]-triangle[1,i]
    
    for i in range(3):
        for j in range(3):
            triangle[i,j]=triangle[i,j]+k*c[j]  
     
#    print(time.time()-time1)       
    if equal(crossing(a,point-triangle[0,:]),-c)==True:
        length2=length_to_edge(triangle[0,:],triangle[1,:],point)
    elif equal(crossing(d,point-triangle[1,:]),-c)==True:
        length2=length_to_edge(triangle[1,:],triangle[2,:],point)
    elif equal(crossing(-b,point-triangle[2,:]),-c)==True:
        length2=length_to_edge(triangle[0,:],triangle[2,:],point)
    else:
        length2=0 
#    print(length1,length2)    
#    print(time.time()-time1)    
    return np.sqrt(length1*length1+length2) 

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













@nb.jit(nopython=True)
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


