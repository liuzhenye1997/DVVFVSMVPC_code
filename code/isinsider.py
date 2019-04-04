# Script to check if a point lies inside or outside a closed STL file
#import sys
#sys.path.append('/data_b/liuzhenye/test1/vtk/')
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import length
import length_temp
import length_torch
import numpy 
import torch
import numba as nb


#@nb.jit(nopython=True)
#def label_point(number,distance,pdata_GetNumberOfPoints,obj_points,points,labels):
#    for i in range(number): 
#        distance[i]=length.length_to_point(pdata_GetNumberOfPoints,obj_points,points[i])
#        if distance[i]<1:
#            labels[i,0]=labels[i,1]=1

#判断空间点是否在模型表面
#@nb.jit
def label_surface(obj_polygons,pdata_GetNumberOfCells,distance,number,labels,points):
#    length_torch=length_temp.Length(pdata_GetNumberOfCells)

#    for i in range(number):
    obj_polygons_temp=torch.tensor(obj_polygons.copy())
    time2=time.time()   
   
#       
    
    distance=length_temp.length(pdata_GetNumberOfCells,obj_polygons_temp,points)
    
    
#        print(pdata.GetNumberOfCells())
#        print(time.time()-time2)
#        distance[i]=length.length_to_point(pdata.GetNumberOfPoints(),obj_points,numpy.array(points[i]))
    for i in range(number):
        if distance[i]<1:
            labels[i,0]=labels[i,1]=1
#            print(distance[i])
    
            
#create a points polydata
def getPolydata(i,j,k):
    
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(i,j,k) # selected a point here which lies inside the stl surface , Bounds of the stl file are (-0.5, -0.5, -0.5, 0.5, 0.5, 0)
    pts_pd = vtk.vtkPolyData()
    pts_pd.SetPoints(pts)
    return pts_pd




#判断空间点相对模型的位置
def inside(path,number,points,labels):
#    time1=time.time()
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
    openEdges = featureEdge.GetOutput().GetNumberOfCells()
    
    if openEdges != 0:
        print("STL file is not closed")
        print(openEdges)
        return openEdges
#    print(openEdges)
#    print(pdata.GetNumberOfPoints())
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
#    print(pdata)
    
#    for i in range() 
    # checking for consistency of IsInside method
 
    filter.SetTolerance(0.00001)
    
    time1=time.time()
#    obj_points=numpy.zeros([pdata.GetNumberOfPoints(),3])
#    for i in range(pdata.GetNumberOfPoints()): 
#        obj_points[i,:]=pdata.GetPoint(i)
#    
    obj_polygons=numpy.zeros([pdata.GetNumberOfCells(),3,3])
    for i in range(pdata.GetNumberOfCells()): 
        cell=vtk.vtkIdList()
        pdata.GetCellPoints(i,cell)
        for j in range(3):
            obj_polygons[i,j,:]=pdata.GetPoint(cell.GetId(j))
            
#    print(obj_polygons)    
#    
    distance=numpy.zeros(number)
#    pdata_GetNumberOfPoints=pdata.GetNumberOfPoints()
#    points=numpy.array(points)
#    label_point(number,distance,pdata_GetNumberOfPoints,obj_points,points,labels)
    for i in range(number):  
        filter.SetInputData(getPolydata(points[i,0],points[i,1],points[i,2]))
        filter.Update()            
        labels[i,0]=filter.IsInside(0)
#            if labels[i,0]==1:
#                print(1111,distance[i])
        labels[i,1]=1-labels[i,0] 
      
    pdata_GetNumberOfCells=pdata.GetNumberOfCells()
    label_surface(obj_polygons,pdata_GetNumberOfCells,distance,number,labels,points)    
    
                         
    print(time.time()-time1) 
#    print(distance) 
    return 0

#for j in range(2,-1,-1):           
#    time1=time.time()         
#    number=48
#    #print(number)
#    points=torch.rand([number,3])
#    #points=numpy.array([[-4,0,0]])
#    ##print(points)
#    ##
#    for i in range(number):
#        points[i,:]=torch.tensor([5,255+i,80])
#    labels=torch.zeros([number,2])
#    labels=numpy.zeros([number,2])
#    path = "D:/data/SURFACE/model_"+str(19+j)+"/model_"+str(19+j)+"_anim_"+str(0)+"/00"+str(0)+"1.obj"
#    ##print(path)
#    
#    inside(path,number,points,labels) 
#    print(labels)
#    print(time.time()-time1)
#











#保证模型内部和外部的点一样多
def divide(points,labels):
    number_in=0
    number_out=0
    number_surface=0
#    print(points.shape)
    for i in range(points.shape[0]):
        if labels[i,0]==1 and labels[i,1]==0:
            number_in+=1
        elif labels[i,1]==1 and labels[i,0]==0:
            number_out+=1
        else:
            number_surface+=1
        
    if  number_in>number_out:  
        print('number_in>number_out')
        divide_point=numpy.zeros([2*number_out+number_surface,3])
        divide_labels=numpy.zeros([2*number_out+number_surface,2])
#        print(divide_point)
        j=0
        for i in range(points.shape[0]):
            if labels[i,1]==1 and labels[i,0]==0:
                divide_point[2*j+1,:]=points[i]
                divide_labels[2*j+1,:]=labels[i]
                j=j+1
                
        for i in range(points.shape[0]):
            
            if labels[i,0]==1 and labels[i,1]==0:
                divide_point[2*j,:]=points[i]
                divide_labels[2*j,:]=labels[i]
                j=j+1
                if j==number_out:
                    k=0
                    for i in range(points.shape[0]):                                                
                        if labels[i,0]==1 and labels[i,1]==1:
                            divide_labels[2*number_out+k,:]=labels[i]
                            divide_point[2*number_out+k,:]=points[i]
                            k=k+1
                    value=[]
#                    print(divide_labels)
                    value.append(divide_point)
                    value.append(divide_labels)
                    return value  
    else:
       
        divide_point=numpy.zeros([2*number_in+number_surface,3])
        divide_labels=numpy.zeros([2*number_in+number_surface,2])
#        print(divide_point)
        j=0
        for i in range(points.shape[0]):
            if labels[i,0]==1 and labels[i,1]==0:
                divide_point[2*j+0,:]=points[i]
                divide_labels[2*j+0,:]=labels[i]
                j=j+1
        j=0        
        for i in range(points.shape[0]):
            if labels[i,1]==1 and labels[i,0]==0:
                divide_point[2*j+1,:]=points[i]
                divide_labels[2*j+1,:]=labels[i]
#                print(divide_labels[2*j+1,:])
                j=j+1
                if j==number_in:
                    k=0
                    for i in range(points.shape[0]):                        
#                        print(i)
                        if labels[i,0]==1 and labels[i,1]==1:
#                            print(i)
                            divide_labels[2*number_in+k,:]=labels[i]
#                            print(divide_labels[2*number_in+k,:],labels[i])
                            divide_point[2*number_in+k,:]=points[i]
#                            print(divide_labels[2*number_in+k,:],labels[i])
                            k=k+1
#                            print('k',k)
#                    print(divide_labels)        
                    value=[]
#                    print(divide_labels)
                    value.append(divide_point)
                    value.append(divide_labels)
                    return value 




