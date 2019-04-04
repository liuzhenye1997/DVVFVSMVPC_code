# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:28:47 2018

@author: liuzhenye
"""
import numpy
import vtk
import numba as nb
#计算空间中的点在照片上的位置
@nb.jit
def projection(points,pamt,number):
    point=numpy.ones([1,number])
#    print(point)
    point=numpy.vstack((numpy.transpose(points),point))
#    print(point,pamt)
    point=numpy.dot(pamt,point) 
#    print(point.shape)
    point=numpy.transpose(point)
#    print(point)
    for i in range(number):
        if point[i,2]!=0:
            point[i,:]=point[i,:]/point[i,2]  
#    print(point)            
    return point[:,0:2]

#计算模型的bounding box
def bounding_box(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)    
    reader.Update()
    pdata = reader.GetOutput()
    return pdata.GetBounds()
   

#data = numpy.loadtxt("pmat_0001.txt")
#print(data)

#points=256*numpy.random.rand(4,3)
#number=4
#print(points)
#print(bounding_box('0001.obj'))    

#data = numpy.loadtxt("Camera1.Pmat.txt")
#print(data)
#max=bounding_box('0001.obj')
#print(max)
