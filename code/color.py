# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:03:22 2018

@author: liuzhenye
"""
import projection
import numpy as np
import qr
import math
import vtk
import torch
import matplotlib.pyplot as plt
import obj
import time
import numba as nb

#对模型上某一点染色
@nb.jit(nopython=True)
def color_points_color(points,projection_points,color,weight):
    for i in range(points.shape[0]):
        for j in range(pamt.shape[0]):   
            projection_x=projection_points[j,i,0]
            projection_y=projection_points[j,i,1]
            x_ceil=math.ceil(projection_x)
            x_floor=x_ceil-1
            y_ceil=math.ceil(projection_y)
            y_floor=y_ceil-1 

            y=(photo[j,:,x_ceil,y_ceil]*(x_floor-projection_x)*(y_floor-projection_y)
                    +photo[j,:,x_floor,y_ceil]*(projection_x-x_ceil)*(y_floor-projection_y)
                    +photo[j,:,x_ceil,y_floor]*(x_floor-projection_x)*(projection_y-y_ceil)
                    +photo[j,:,x_floor,y_floor]*(projection_x-x_ceil)*(projection_y-y_ceil))      

            color[i,:]=color[i,:]+weight[i,j]*y*255

#对模型染色    
@nb.jit
def color_points(photo,points,pamt,normals):
    np.save('points',points)
    site=camera_site(pamt.copy())
    np.savetxt('camera_site.txt',site)    
    color=np.zeros((points.shape[0],3))
    direction=np.zeros((points.shape[0],pamt.shape[0],3))
    for i in range(points.shape[0]):
        for j in range(pamt.shape[0]):
            direction[i,j,:]=site[j,:]-points[i,:]
    projection_points=np.zeros((pamt.shape[0],points.shape[0],2))
#    print(pamt.shape[0])
    for i in range(pamt.shape[0]):     
        projection_points[i,:,:]=projection.projection(points,pamt[i,:,:],points.shape[0])
    weight=np.zeros((points.shape[0],pamt.shape[0])) 
    
    for i in range(points.shape[0]): 
        sum=0
        for j in range(pamt.shape[0]):  
#             print(normals[i,:],direction[i,j,:])
             if np.dot(normals[i,:],normals[i,:])==0:
#                 print(111111)
                 weight[i,j]=0
             else:
                 weight[i,j]=np.dot(normals[i,:],direction[i,j,:])/np.sqrt(np.dot(normals[i,:],normals[i,:]))/np.sqrt(np.dot(direction[i,j,:],direction[i,j,:]))
             if weight[i,j]<0:
                 weight[i,j]=0
             sum=sum+weight[i,j]   
            

        for j in range(pamt.shape[0]):
            if sum>0:
                weight[i,j]=weight[i,j]/sum
            else :
                weight[i,j]=0     
              
    color_points_color(points,projection_points,color,weight)         
#    print(OK)        
    np.save('color_1',color)          
    return color    

#计算相机在世界坐标系的坐标
def camera_site(pamt):          
    number_of_camera=pamt.shape[0]
    site=np.zeros([number_of_camera,3])
    for i in range(number_of_camera):         
        (R,Q)=qr.RQ(pamt[i,:,0:3].copy())
        T=np.dot(np.linalg.inv(R),pamt[i,:,:])

        site[i,:]=-np.dot(Q.transpose(),T[:,3])
    np.savetxt('camera_site.txt',site) 
    return site 

#读取obj模型
def vtk_read(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    
    reader.Update()
    pdata = reader.GetOutput()
    
    # check if the stl file is closed
    featureEdge = vtk.vtkFeatureEdges()
    featureEdge.FeatureEdgesOff()
    featureEdge.BoundaryEdgesOn()
    featureEdge.NonManifoldEdgesOn()
    featureEdge.SetInputData(pdata)
    featureEdge.Update()

    

    # pass pdata through a triangle filter
    tr= vtk.vtkTriangleFilter()
    tr.SetInputData(pdata)
    tr.PassVertsOff()
    tr.PassLinesOff()
    tr.Update()
    
    # normals filter
    pnormal = vtk.vtkPolyDataNormals()
    pnormal.SetInputData(tr.GetOutput())
    pnormal.AutoOrientNormalsOff()
    pnormal.ComputePointNormalsOn()
    pnormal.ComputeCellNormalsOff() 
    pnormal.SplittingOff()
    pnormal.ConsistencyOn()
    pnormal.FlipNormalsOn()
    pnormal.Update()
    pdata = pnormal.GetOutput()
    
    # create a vtkSelectEnclosedPoints filter
    filter = vtk.vtkSelectEnclosedPoints()
    filter.SetSurfaceData(pdata)
    print(pdata.GetNumberOfPoints()) 
    print(pdata.GetPointData().GetNumberOfTuples())
    print(pdata)
    obj_points=np.zeros([pdata.GetNumberOfPoints(),3])
    obj_normals=np.zeros([pdata.GetNumberOfPoints(),3])
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(pdata)
    
    for i in range(pdata.GetNumberOfPoints()): 
        obj_points[i,:]=pdata.GetPoint(i)  
        obj_normals[i,:]=pdata.GetPointData().GetNormals().GetTuple(i)
    return (obj_points,obj_normals)    

camera=np.array([1,3,5,7])  
number_of_camera=4
pamt=np.zeros([number_of_camera,3,4])
for i in range(number_of_camera):
    pamt[i,:,:] = np.loadtxt("Camera"+str(camera[i])+".Pmat.cal")      
#    print(pamt[i,:,:],'\n')


site=camera_site(pamt.copy())



model=19
number_fo_anim=0
number_of_photo=0
#path = "H:/data/OBJ/model_"+str(model)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/00"+str(number_of_photo)+"1.obj"
path=r"C:/Users/liuzhenye/Desktop/code/test15/13.obj"
#


(obj_points,obj_normals)=vtk_read(path)    
                 

photo=np.array(torch.empty(number_of_camera,3,1600,1200))
for i in range(number_of_camera):                    
    photo[i,:]=np.transpose(plt.imread("G:/data/RENDER/anim_"+str(number_fo_anim)+"/model_"+str(model)+"_anim_"+str(number_fo_anim)+"/cameras_cam0"+str(camera[i])+"/alpha_00"+str(number_of_photo)+"1.png")).astype(float)
    plt.imshow(np.transpose(photo[i,:]))
    plt.show()
print(np.max(obj_points[:,1]))   



bouding_box=projection.bounding_box(path)
print(bouding_box)               
    
time1=time.time()    
color_points(photo,obj_points,pamt,obj_normals)
print(time.time()-time1)
np.save('points_1',obj_points)         