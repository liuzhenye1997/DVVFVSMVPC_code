# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:03:44 2018

@author: liuzhenye
"""
import numpy as np
import numba as nb
#用于读取并处理obj模型，但实际上大多时候使用的是vtk库



@nb.jit(nopython=True)
def crossing(a,b):
    c=np.zeros(3)
    c[0]=a[1]*b[2]-a[2]*b[1]
    c[1]=a[2]*b[0]-a[0]*b[2]
    c[2]=a[0]*b[1]-a[1]*b[0]
    return c

class OBJ:
    #读取obj模型
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl=None

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
           
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
    #计算模型边界            
    def boundary(self):
        x_list=[]
        y_list=[]
        z_list=[]
        for i in range(len(self.vertices)):
            x_list.append(self.vertices[i][0])
            y_list.append(self.vertices[i][1])
            z_list.append(self.vertices[i][2])    
        x_list.sort()
        y_list.sort()
        z_list.sort()
        boundary=np.zeros((3,2))
        
        boundary[0,0]=x_list[int(0.01*len(self.vertices))]
        boundary[0,1]=x_list[int(0.99*len(self.vertices))]
        boundary[1,0]=y_list[int(0.01*len(self.vertices))]
        boundary[1,1]=y_list[int(0.99*len(self.vertices))]
        boundary[2,0]=z_list[int(0.01*len(self.vertices))]
        boundary[2,1]=z_list[int(0.99*len(self.vertices))]        
        return boundary

    #计算模型的中心点，主要是为了找一个在模型的内部点。但这种方法总是不稳定的。这也是为何使用vtk库的主要原因。
    def Center(self):
        boundary=self.boundary()
        center=np.zeros(3)
        center=(boundary[:,0]+boundary[:,1])/2
        return center
    #计算模型上的面的法向量
    def Face_normal(self):
    
        center=self.Center()
        face_normal=np.zeros((len(self.faces),3))
        for i in range(len(self.faces)):
            print(self.faces[i][1])
            a=self.vertices[self.faces[i][2]]-self.vertices[self.faces[i][1]]
            b=self.vertices[self.faces[i][2]]-self.vertices[self.faces[i][0]]
            c=crossing(a,b)
            face_center=(self.vertices[self.faces[2],:]+self.vertices[self.faces[1],:]+self.vertices[self.faces[0],:])/3
            if np.dot(c,face_center-center)>0:
                face_normal[i,:]=c
            else:
                face_normal[i,:]=-c
        return face_normal        
    #计算模型上的点的法向量            
    def point_normal(self):
        face_normal=self.Face_normal()
        number_of_vertices=np.zeros(len(self.vectices))
        for i in range(len(self.faces)):
            for j in range(3):
                number_of_vertices[self.faces[i,j]]=number_of_vertices[self.faces[i,j]]+1
        max_=np.max(number_of_vertices)
        point_list=np.zeros((len(self.vectices),max_))
        number_of_vertices=np.zeros(len(self.vectices))
        for i in range(len(self.faces)):
            for j in range(3): 
                point_list[self.faces[i,j],number_of_vertices[self.faces[i,j]]]=i
                number_of_vertices[self.faces[i,j]]=number_of_vertices[self.faces[i,j]]+1
        point_normal= np.zeros((len(self.vertices),3))       
        for i in range(len(self.vertices)):
            for j in range(number_of_vertices[i]):
                point_normal[i,:]=point_normal[i,:]+face_normal[point_list[i,j],:]
            point_normal[i,:]=point_normal[i,:]/np.dot(point_normal[i,:],point_normal[i,:])   
        return point_normal        
            
        
                        
                
                
                
                

# In[]
obj=OBJ('12.obj')
obj.point_normal()
#print(len(obj.faces))
#np.savetxt('points_1.txt',obj.vertices) 