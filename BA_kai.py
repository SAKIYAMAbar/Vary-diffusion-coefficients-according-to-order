import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from numba import jit
import seaborn as sns
import pandas as pd
import random

@jit
def make_d(d,A,N):#次数行列を作成
    for i in range(N):
        for j in range(N):
            if A[i,j]==1:
                d[i,i]=d[i,i]+1


@jit
def max_d(d,N):#最大のノード次数を取得
    #max=0
    min=N
    index=0
    for i in range(N):
        #if d[i][i]>max:
            #max=d[i][i]
            #index=i
        if d[i][i]<min:
            min=d[i][i]
            index=i
    print("index",index)
    #print("max",d[index][index])
    print("min",d[index][index])
    return index  


@jit
def adj(N,A,u,index):#隣接行列に初期値を与える
    #for i in range(N):
        #if A[index,i]==1 or i==index:
            #u[i]=0.1
    u[index]=0.1

@jit
def laplacian(s,L):#ラプラシアンを求める
    L1=s.shape
    S=int(s.size)
    ts = np.zeros(S)
    for i in range(S):
        for j in range(S):
            #ts[i]+=L[j,i]*s[j]
            ts[i]+=L[i,j]*s[j]
    return ts


@jit
def calc(a, h, a2, h2, La,c,u,v):#状態量を求める
    L = a.size
    (L2,L2)=La.shape
    heikin=0
    dt=0.01
    #Dh=0.5#パラメーター始
    #Dh=0.5#パラメーター始
    Dh=0.5
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=(da+mua-ca)/10
    #roh=(muh-ch)/10
    roa=np.zeros(L)
    roh=np.zeros(L)
    for i  in range(L):
        #roa[i]=(mua-ca)*u[i]+da*v[i]
        #roh[i]=(-ch)*u[i]+muh*v[i]
        roa[i]=(mua-ca+da)*(0.01*La[i,i])
        roh[i]=(-ch+muh)*(0.01*La[i,i])
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    #scale=(-Dh*fa/gh)/((Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh))
    if c==0:
        Da=0
        Dh=0
    if c==1:
        Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    elif c==2:
        Da=0.2
    elif c==3:
        Da=0.02
    DA=np.zeros(L)
    
    La1=np.zeros((L,L))
    kappa_a=np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            La1[i][j]=La[i,j]/La[j,j]
            #La1[i][j]=La[i,j]/(np.sqrt(La[j,j]*La[i,i]))
            #La1[i][j]=La[i,j]/La[i,i]
            #La1[i][j]=La[i,j]
    mina=0
    minh=0
    maxa=1
    maxh=1 
        #sa = ((ca*a)-(da*h)+roa-mua*a - Da*laplacian(a,kappa_a))*dt ##反応項と拡散項を計算
    sa = ((ca*a)-(da*h)+roa-mua*a -Da * laplacian(a,La1))*dt ##反応項と拡散項を計算
    sh = ((ch*a)+roh-muh*h -Dh * laplacian(h,La1))*dt  
    for i in range(L):
            a2[i] = a[i]+(sa[i]) #-mua*a[i,j]
            h2[i] = h[i]+(sh[i]) # -muh*h[i,j]           
            if a2[i]<mina:
                a2[i]=mina
            if h2[i]<minh:
                h2[i]=minh
            if a2[i]>maxa:
                a2[i]=maxa
            if h2[i]>maxh:
                h2[i]=maxh


def pic(N,u,v,G,indexlist,us,vs):#図示する
    judgeu=0
    judgev=0
    u0=np.zeros(N)
    v0=np.zeros(N)
    for j in range(N):
        u[j]=round(u[j],2)
        v[j]=round(v[j],2)  
        u0[j]=round(us[j],2)
        v0[j]=round(vs[j],2) 
        if u[j]!=u0[j]:
            judgeu=1
    if judgeu==0:
        print("correct")
    else:
        print("non correct")
    print("maxu",np.max(u),"minu",np.min(u),"maxv",np.max(v),"minv",np.min(v),"maxu0",np.max(u0),"minu0",np.min(u0),"maxv0",np.max(v0),"minv0",np.min(v0))
    fig, ax = plt.subplots()
    d_u=np.sort(u)
    d_v=np.sort(v)
    ax.set_title('activator')
    for i in range(N):
      plt.scatter(indexlist[i], d_u[i],c="red",s=10)
      #plt.scatter(indexlist[i], u0[i],c="black",s=10)
    ax.set_xlabel("Node ID")
    ax.set_ylabel("u") 
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title('inhibitor')
    for i in range(N):
        plt.scatter(indexlist[i], d_v[i],c="blue",s=10)
        #plt.scatter(indexlist[i], v0[i],c="black",s=10)
    ax.set_xlabel("Node ID")
    ax.set_ylabel("v") 
    plt.show()


def main():
    N = 200# the number of points
    indexlist=np.zeros(N)
    d=np.zeros((N,N))
    A=np.zeros((N, N))
    for i in range(N):
      indexlist[i]=i
    np.random.seed(seed=0)
    #レギュラーグラフ
    #################################################
    #G = nx.random_regular_graph(4, N, seed=0)#レギュラーグラフ
    #G=nx.erdos_renyi_graph(N, 0.02,seed=33)
    G=nx.barabasi_albert_graph(N,2 ,seed=0)
    pos = nx.spring_layout(G)
    A = nx.to_numpy_matrix(G)
    ############################################               
    make_d(d,A,N)
    L=d-A#ラプラシアン行列
    #h=max_d(d,N)
    h=N//2
    u0= np.zeros(N)+0.1
    adj(N,A,u0,h)
    u02 =np.zeros(N) 
    v0 = np.zeros(N)+0.1
    v02 =np.zeros(N)
    uc0 =np.zeros(N) 
    vc0 =np.zeros(N)
    random.seed(0)
    for i in range(N):#初期状態をランダムに設定
            u0[i] = 0.01*(L[i,i])
            v0[i] = 0.01*(L[i,i])
    plt.subplot()
    plt.figure(figsize=(6,4))
    nx.draw(G, node_size=20)
    nx.draw(G, node_size=20)
    plt.tight_layout()
    plt.show()
    time=100000
    for k in range(4):
        a=np.zeros(N)
        a2=np.zeros(N)
        h=np.zeros(N)
        h2=np.zeros(N)
        for i in range(N):#これができないと配列が初期化できない
                a[i]=u0[i]
                a2[i]=u02[i]
                h[i]=v0[i]
                h2[i]=v02[i]
                #max_index=np.argmax(u0)
                #a[max_index]=u0[max_index]+0.01
                min_index=np.argmin(u0)
                a[min_index]=u0[min_index]+0.01
               
        for i in range(time):
            if i % 2 == 0:
                    calc(a, h, a2, h2, L,k,uc0,vc0)
            else:                  
                    calc(a2, h2, a, h, L,k,uc0,vc0)
                    #現在のステップの状態u2,v2から次のステップの状態u,vを計算する
            if i==0 and k==0:   
                    pic(N,a,h,G,indexlist,u0,v0) 
            if i==time-1:   
                    pic(N,a,h,G,indexlist,u0,v0) 
                        
main=main()
