# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:28:35 2018

@author: liuzhenye
"""

#coding:utf8
import numpy as np

def gram_schmidt(A):
    """Gram-schmidt正交化"""
    Q=np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) # 减去待求向量在以求向量上的投影
        e = u / np.linalg.norm(u)  # 归一化
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return (Q, R)

def givens_rotation(A):
    """Givens变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
            r_ = np.hypot(R[col, col], R[row, col])  # d
            c = R[col, col]/r_
            s = -R[row, col]/r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
    return (Q, R)

def householder_reflection(A):
    """Householder变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
    return (Q, R)

#RQ分解，用于分解相机参数矩阵
def RQ(M):
    M_flipud=M.copy()
#    print(M)
    for i in range(M.shape[0]):
        M[i,:]=M_flipud[M.shape[0]-1-i,:]
        
#    print(M)    
    (Q, R) = householder_reflection(M.transpose())

   
    R_flipud=R.transpose().copy()
    R=R.transpose()
    for i in range(R.shape[0]):
        R[i,:]=R_flipud[R.shape[0]-1-i,:]
#        print(R_flipud)
#        
    Q=Q.transpose()
    R_fliplr=np.array(R).copy()
    for i in range(R.shape[1]):
  
        R[:,i]=R_fliplr[:,R.shape[1]-1-i]
#       
#
    Q_flipud=np.zeros([Q.shape[0],Q.shape[1]])
    Q_flipud=Q.copy()
    for i in range(Q.shape[0]):
        Q[i,:]=Q_flipud[Q.shape[0]-1-i,:]
    return (R,Q)

#np.set_printoptions(precision=4, suppress=True)
#A = np.array([[6, 5, 0],[5, -1, 4],[5, 1, -14],[0, 4, 3]],dtype=float)
#
#(Q, R) = gram_schmidt(A)
#print(Q)
#print(R)
#print (np.dot(Q,R))
#
#(Q, R) = givens_rotation(A)
#print(Q)
#print(R)
#print (np.dot(Q,R))
#
#(Q, R) = householder_reflection(A)
#print(Q)
#print(R)
#print(np.linalg.det(Q[0:3,0:3]))
#print (np.dot(Q,R))
#
#print(A.transpose())
#(R,Q)=RQ(A.transpose().copy())
##print(R)
##print(Q)
#print (np.dot(R,Q))

