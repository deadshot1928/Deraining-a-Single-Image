#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2 as cv
import numpy as np
import math


# In[10]:


def wlq(k,yq,yp):
    return 1/(1+np.exp(k*(yp-yq)))


# In[11]:


def wdq(px,py,qx,qy):
    x=(px-qx)*(px-qx)
    y=(py-qy)*(py-qy)
    xy=x+y
    return np.exp(-1*(xy/9))


# In[12]:


def wcq(I):
    return np.exp(-1*I/81)


# In[13]:


def Calculate(img,wx,wy):
    r=img.shape[0]
    c=img.shape[1]
    wt = [[0.1]*c]*r
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    pb=img[wx][wy][0]
    pg=img[wx][wy][1]
    pr=img[wx][wy][2]
    
    yp=0
    for i in range(wx-4,wx+5):
        for j in range(wy-4,wy+5):
            if(i<0 or j<0 or i>=r or j>=c):
                continue
                
            yp+=gray[i][j]
    yp=yp/81
    
    for i in range(wx-4,wx+5):
        for j in range(wy-4,wy+5):
            if(i<0 or j<0 or i>=r or j>=c):
                continue
            yq=gray[i][j]
            w1=wlq(0.1,yq,yp)
            
            w2=wdq(wx,wy,i,j)
            
            qb=img[i][j][0]
            qg=img[i][j][1]
            qr=img[i][j][2]
            I=(pb-qb)*(pb-qb)+(pg-qg)*(pg-qg)+(pr-qr)*(pr-qr)
            w3=wcq(I)
            
            w=w1*w2*w3
            
            wt[i][j]=w
    return wt
            


# In[14]:


def gradx(img,wx,wy):
    r,c,dummy=img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    gx = [[0]*c]*r
    
    
    for i in range(wx-4,wx+5):
        for j in range(wy-4,wy+5):
            if(i<0 or j<0 or i>=r or j>=c):
                continue
            
            m = gray[i-1][j-1]*(-1) if (i-1>=0 and j-1>=0) else 0 
            n = gray[i-1][j]*(-2) if (i-1>=0) else 0
            p = gray[i-1][j+1]*(-1) if (i-1>=0 and j+1<c) else 0
            d = gray[i+1][j-1] if (i+1<r and j-1>=0) else 0
            e = gray[i+1][j]*(2) if (i+1<r) else 0
            f = gray[i+1][j+1] if (i+1<r and j+1<c) else 0
            
            val = m + n + p + d + e + f
            gx[i][j]=val
    
    return gx
            
            


# In[15]:


def grady(img,wx,wy):
    r,c,dummy=img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    gy = [[0]*c]*r
    
    
    for i in range(wx-4,wx+5):
        for j in range(wy-4,wy+5):
            if(i<0 or j<0 or i>=r or j>=c):
                continue
            
            m = gray[i-1][j-1]*(-1) if (i-1>=0 and j-1>=0) else 0 
            n = gray[i][j-1]*(-2) if (j-1>=0) else 0
            p = gray[i+1][j-1]*(-1) if (i+1<r and j-1>=0) else 0
            d = gray[i-1][j+1] if (i-1>=0 and j+1<c) else 0
            e = gray[i][j+1]*(2) if (j+1<c) else 0
            f = gray[i+1][j+1] if (i+1<r and j+1<c) else 0
            
            val = m + n + p + d + e + f
            gy[i][j]=val
    
    return gy


# In[16]:



def Detect(img):
    r,c,dummy=img.shape
    M = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    wsize=9
    f=1
    for i in range(0,r):
        for j in range(0,c):
            wx=i
            wy=j 
            wt=Calculate(img,wx,wy)
            gx=gradx(img,wx,wy)
            gy=grady(img,wx,wy)
            
            COV = [[0]*2]*2
            temp = [[0]*2]*2
            for m in range(wx-4,wx+5):
                for n in range(wy-4,wy+5):
                    if(m<0 or n<0 or m>=r or n>=c):
                        continue
                    COV[0][0]+=(wt[m][n]*wt[m][n]*gx[m][n]*gx[m][n])
                    COV[0][1]+=(wt[m][n]*wt[m][n]*gx[m][n]*gy[m][n])
                    COV[1][0]+=(wt[m][n]*wt[m][n]*gx[m][n]*gy[m][n])
                    COV[1][1]+=(wt[m][n]*wt[m][n]*gy[m][n]*gy[m][n])
                    
            Z=0
            for m in range(wx-4,wx+5):
                for n in range(wy-4,wy+5):
                    if(m<0 or n<0 or m>=r or n>=c):
                        continue
                    Z+=wt[m][n]
                        
            COV[0][0]/=Z
            COV[0][1]/=Z
            COV[1][0]/=Z
            COV[1][1]/=Z
            U, S, vh = np.linalg.svd(COV, full_matrices=True)
            sin=U[1][0]
            lda=S[0]
            mu=S[1]
            
            if(sin>0.8660254037 and lda/mu>2 and mu>10):
                M[i][j]=1
            else:
                M[i][j]=0
            
    return M
                    


# In[23]:


def Remove(img,M):
    r,c=M.shape
    N=0
    for i in range(0,r):
        for j in range(0,c):
            Bp=[[0]*15]*1
            Rp=[[0]*15]*1
            for m in range(i-7,i+8):
                if(i<0 or i>=r or m<0 or m>=r):
                    if(m+7-i<0 or m+7-i>=15):
                        continue
                    Bp[m+7-i][0]=0
                    Rp[m+7-i][0]=0
                else:
                    Bp[m+7-i][0]=img[m][j]
                    Rp[m+7-i][0]=M[m][j]
            tb=0
            tg=0
            tr=0
            
            for m in range(i-25,i+26):
                for n in range(j-25,j+26):
                    if(i<0 or j<0 or i>=r or j>=c):
                        continue
                    Bq=[[0]*15]*1
                    Rq=[[0]*15]*1
                    for q in range(i-7,i+8):
                        if(i<0 or i>=r or q<=0 or q>=r):
                            Bq[q+7-i][0]=0
                            Rq[q+7-i][0]=0
                        else:
                            Bq[q+7-i][0]=img[q][j]
                            Rq[q+7-i][0]=M[q][j]
                    N=0
                    for q in range(0,15):
                        t=(1-Rp[q][0])*(1-Rq[q][0])
                        if(t==1):
                            N+=1
                        Bp[q][0]=Bp[q][0]*t
                        Bq[q][0]=Bq[q][0]*t
                    
                    Bpq=0
                    for q in range(0,15):
                        Bpq+=(Bp[q][0]-Bq[q][0])*(Bp[q][0]-Bq[q][0])
                        
                    tb+=(np.exp(-1*Bpq/(225*N))*(1-M[i][j])*img[i][j][0])/(np.exp(-1*Bpq/(225*N))*(1-M[i][j]))
                    tg+=(np.exp(-1*Bpq/(225*N))*(1-M[i][j])*img[i][j][1])/(np.exp(-1*Bpq/(225*N))*(1-M[i][j]))
                    tr+=(np.exp(-1*Bpq/(225*N))*(1-M[i][j])*img[i][j][2])/(np.exp(-1*Bpq/(225*N))*(1-M[i][j]))
            
            img[i][j][0]=tb
            img[i][j][1]=tg
            img[i][j][2]=tr
            
    
    return img
                    
                    
            
            
    
    


# In[26]:




img=cv.imread("C:/Users/USER/Desktop/vc.jpg")

M=Detect(img)

#out=Remove(img,M)

cv.imshow("Map",M)
cv.waitKey(0)
        
    
    


# In[ ]:




