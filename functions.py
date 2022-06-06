# -*- coding: utf-8 -*-

import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import shapely.geometry as geometry

from scipy.spatial import ConvexHull

import os

from sklearn.neighbors import KNeighborsClassifier


def layer_function(data,n):
    layer=data[(data.Cota<=n+0) & (data.Cota>=n-0)] 
    gravels=layer[layer.Clase=='gravillas y gravas']
    sands=layer[layer.Clase=='arenas']
    clays=layer[layer.Clase=='arcillas y limos']
    basement=layer[layer.Valor=='S']
    return [layer,gravels,sands,clays,basement]

def zip_xyz(dat):
    Xnn=np.concatenate((np.array(dat[1]['UTM_X']),np.array(dat[2]['UTM_X']),
                      np.array(dat[3]['UTM_X']),np.array(dat[4]['UTM_X'])))

    Ynn=np.concatenate((np.array(dat[1]['UTM_Y']),np.array(dat[2]['UTM_Y']),
                      np.array(dat[3]['UTM_Y']),np.array(dat[4]['UTM_Y'])))
    Znn=np.concatenate((np.array(dat[1]['Cota']),np.array(dat[2]['Cota']),
                      np.array(dat[3]['Cota']),np.array(dat[4]['Cota'])))
    return list(zip(Xnn,Ynn,Znn))

def knn(data,contourn,height,ax,grid):
    xx, yy = np.meshgrid(np.linspace(ax[0],ax[1],grid),np.linspace(ax[2],ax[3],grid))
    C=np.zeros(xx.shape,dtype = float) # Here we save the predictions
    outside=[] # Outside the contourn (not draw)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if contourn.distance(geometry.Point(xx[i,j],yy[i,j]))>1:
                outside.append((i,j))  
    
    df=layer_function(data,height)
    # 0 gravels, 1 sands, 2 clays, 3 basement
    clas=[0]*len(df[1])+[1]*len(df[2])+[2]*len(df[3])+[3]*len(df[4]) # Labels of the known point

    dat=zip_xyz(df)
    
    classifier=KNeighborsClassifier(1,weights='distance')
    classifier.fit(dat,clas) # KNN reads the data
    for i in range(xx.shape[0]):
        d=list(zip(xx[i],yy[i],[height]*len(xx[i])))
        C[i]=classifier.predict(d) # KNN predice para los puntos de la cuadrícula usando los puntos conocidos
    for (i,j) in outside:
                C[i,j]=np.nan # Para que no pinte fuera del contorno
    xyc=[xx,yy,C]
    
    return xyc

def xyc(data,cp,ax,cls,height,grid):
    cc=knn(data,cp,height,ax,grid)
    x1=[cc[0][i,j] for i in range(cc[0].shape[0]) for j in range(cc[0].shape[1]) if cc[2][i,j]==cls]
    y1=[cc[1][i,j] for i in range(cc[1].shape[0]) for j in range(cc[1].shape[1]) if cc[2][i,j]==cls]
    z1=[height for i in range(cc[0].shape[0]) for j in range(cc[0].shape[1]) if cc[2][i,j]==cls]
    punt=[x1,y1,z1]
    return punt

def coordinates(data,positions):
    p=positions
    dat=open(data,'r')
    lg=dat.readlines()
    n_line=len(lg)
    x=[]
    y=[]
    z=[]
    for i in range(1,n_line):
        split_line=lg[i].split(",")
        xyz_t=[]
        x.append(float(split_line[p[0]].rstrip()))
        y.append(float(split_line[p[1]].rstrip()))
        z.append(float(split_line[p[2]].rstrip()))
    return [x,y,z]

def bounds(list):
    x_min=min(list[0])
    x_max=max(list[0])
    y_min=min(list[1])
    y_max=max(list[1])
    w=x_max-x_min 
    h=y_max-y_min
    return [x_min,x_max,y_min,y_max,w,h]

def bounds_join(c1,c2):
    return[min(c1[0],c2[0]),
           max(c1[1],c2[1]),
           min(c1[2],c2[2]),
           max(c1[3],c2[3]),
           max(c1[1],c2[1])-min(c1[0],c2[0]),
           max(c1[3],c2[3])-min(c1[2],c2[2])
           ]

def data_p(list,names,colors,symbols,siz):
    n=len(list)
    return [go.Scatter3d(x=list[i][0], y=list[i][1], z=list[i][2],
            mode ='markers',
            name=names[i],
            marker = dict(size = siz,
                          color =colors[i],
                          opacity = 1,
                          symbol=symbols[i])
                        )
          for i in range(n) ]

def data_p_nl(list,names,colors,symbols,siz):
    n=len(list)
    return [go.Scatter3d(x=list[i][0], y=list[i][1], z=list[i][2],
            mode ='markers',
            name=names[i],
            showlegend=False,
            marker = dict(size = siz,
                          color =colors[i],
                          opacity = 1,
                          symbol=symbols[i])
                        )
          for i in range(n) ]

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def within(x,list,n):
    return [y for y in list if distance(x,y)<n]
    

def within2(list1,list2,n):
    if len(list1)==0:
        return list1
    elif len(list2)==0:
        return list2
    else:
        l=[within(x,list2,n) for x in list1]
        return np.unique(np.concatenate(l),axis=0)
    

def grouping(list1,list2,dist):
    newlist1=within2(list1,list2,dist)
    newlist2=within2(newlist1,list2,dist)
    if len(newlist1)==len(newlist2):
        return newlist1
    else:
        return grouping(newlist2,list2,dist)

def lithosome(point,cls,radio,heights,name):
    cluster=[grouping([point],x,radio) for x in cls]
    noempty=[(cluster[i],i) for i in range(len(cluster)) if len(cluster[i])!=0]
    tt=np.concatenate([[np.concatenate([x,[heights[noempty[i][1]]]]) 
                        for x in noempty[i][0]] 
                        for i in range(len(noempty))
                      ])
    ptt=[[x[0] for x in tt],[x[1] for x in tt],[x[2] for x in tt]]
    h = ConvexHull(tt)
    ph=h.points
    vh=h.vertices
    sh=h.simplices
    return [ph,vh,sh,name]

def data_lithosome(vhull,shull,name,alpha,opc,col):
    return go.Mesh3d(x=vhull[:, 0],y=vhull[:, 1], z=vhull[:, 2], 
                        name=name,
                        showlegend=True,
                        colorbar_title=name,
                        color=col, 
                     i=shull[:, 0], j=shull[:, 1], k=shull[:, 2],
                        opacity=opc,
                        alphahull=alpha,
                        showscale=False
                       ) 

#DISTANCE FUNCTION
def dist(x1,y1,x2,y2):
    d=np.sqrt((x1-x2)**2+(y1-y2)**2)
    return d
#CREATING IDW FUNCTION
def idw_npoint(xz,yz,x,y,z,n_point,p):
    r=5 #block radius iteration distance
    nf=0
    while nf<=n_point: #will stop when np reaching at least n_point
        x_block=[]
        y_block=[]
        z_block=[]
        r +=10 # add 10 unit each iteration
        xr_min=xz-r
        xr_max=xz+r
        yr_min=yz-r
        yr_max=yz+r
        for i in range(len(x)):
            # condition to test if a point is within the block
            if ((x[i]>=xr_min and x[i]<=xr_max) and (y[i]>=yr_min and y[i]<=yr_max)):
                x_block.append(x[i])
                y_block.append(y[i])
                z_block.append(z[i])
        nf=len(x_block) #calculate number of point in the block
    
    #calculate weight based on distance and p value
    w_list=[]
    for j in range(len(x_block)):
        d=dist(xz,yz,x_block[j],y_block[j])
        if d>0:
            w=1/(d**p)
            w_list.append(w)
            z0=0
        else:
            w_list.append(0) #if meet this condition, it means d<=0, weight is set to 0
    
    #check if there is 0 in weight list
    w_check=0 in w_list
    if w_check==True:
        idx=w_list.index(0) # find index for weight=0
        z_idw=z_block[idx] # set the value to the current sample value
    else:
        wt=np.transpose(w_list)
        z_idw=np.dot(z_block,wt)/sum(w_list) # idw calculation using dot product
    return z_idw


def interpolation(list_of_points,n,bounds):
    [x_min,x_max,y_min,y_max,w,h]=bounds
    [x,y,z]=list_of_points
    wn=w/n #x interval
    hn=h/n #y interval
    #list to store interpolation point and elevation
    y_init=y_min
    x_init=x_min
    x_idw_list=[]
    y_idw_list=[]
    z_head=[]
    for i in range(n):
        xz=x_init+wn*i
        yz=y_init+hn*i
        y_idw_list.append(yz)
        x_idw_list.append(xz)
        z_idw_list=[]
        for j in range(n):
            xz=x_init+wn*j
            z_idw=idw_npoint(xz,yz,x,y,z,5,1.5) #min. point=5, p=1.5
            z_idw_list.append(z_idw)
        z_head.append(z_idw_list)
    return [z_head,x_idw_list,y_idw_list]

def cutting(list_of_points,polyg,dis):
    pc=list_of_points.copy()
    m=len(pc[1])
    for i in range(m):
        for j in range(m):
            if polyg.distance(geometry.Point(pc[1][i],pc[2][j]))>dis:
                pc[0][j][i]=np.nan
    return pc