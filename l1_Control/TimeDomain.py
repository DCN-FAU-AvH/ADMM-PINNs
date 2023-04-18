#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:42:15 2022

@author: yhr
"""
import numpy as np

class Space1d():
    def __init__(self,a,b):
        self.a=a
        self.b=b

    # def on_boundary(self,xt):
    #     return np.any(np.isclose(xt[:, :-1], [self.a, self.b]), axis=-1)
    # def on_initial(self,xt):
    #     return np.isclose(xt[:, -1:],self.t0).flatten()
    def uniform_points(self,n,boundary=True):
        if boundary:
            x=np.linspace(self.a, self.b, num=n)[:, None]
        else:
            x=np.linspace(self.a, self.b, num=n+1,endpoint=False)[1:, None]
        return x
    def random_points(self,nx):
        xr=np.random.rand(nx)[:, None]
        x=(self.b-self.a)*xr+self.a
        return x
    def uniform_boundary_points(self):
        return np.vstack((self.a, self.b))

class Space1dXTime():
    def __init__(self,a,b,t0,t1):
        self.a=a
        self.b=b
        self.t0=t0
        self.t1=t1

    # def on_boundary(self,xt):
    #     return np.any(np.isclose(xt[:, :-1], [self.a, self.b]), axis=-1)
    # def on_initial(self,xt):
    #     return np.isclose(xt[:, -1:],self.t0).flatten()
    def uniform_points(self,nx,nt,boundary=True):
        if boundary:
            x=np.linspace(self.a, self.b, num=nx)[:, None]
            t=np.linspace(self.t0, self.t1, nt)[:, None]
        else:
            x=np.linspace(self.a, self.b, num=nx+1,endpoint=False)[1:, None]
            t=np.linspace(self.t0, self.t1, num=nt+1,endpoint=False)[1:, None]
        xt=[]
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti))))
        xt = np.vstack(xt)
        return xt
    def random_points(self,nx,nt):
        xr=np.random.rand(nx)[:, None]
        x=(self.b-self.a)*xr+self.a
        tr=np.random.rand(nt)[:, None]
        t=(self.t1-self.t0)*tr+self.t0
        xt=[]
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti))))
        xt = np.vstack(xt)
        return xt
    def uniform_boundary_points(self,n):
        nx=2
        xa = np.full((nx // 2, 1), self.a)
        xb = np.full((nx-nx // 2, 1), self.b)
        x=np.vstack((xa, xb))
        nt = int(np.ceil(n / nx)) 
        t = np.linspace(self.t1,self.t0,num=nt,endpoint=False)
        xt=[]
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti))))
        xt = np.vstack(xt)
        return xt
    def random_boundary_points(self,n):
        tr=np.random.rand(n)[:, None]
        t=(self.t1-self.t0)*tr+self.t0
        x=np.random.choice([self.a, self. b], n)[:, None]
        xt = np.hstack((x,t))
        return xt
    def uniform_initial_points(self, n):
        x=np.linspace(self.a, self.b, num=n)[:, None]
        t = self.t0
        return np.hstack((x, np.full([len(x), 1], t)))
    def uniform_final_points(self, n):
        x=np.linspace(self.a, self.b, num=n)[:, None]
        t = self.t1
        return np.hstack((x, np.full([len(x), 1], t)))
    def random_initial_points(self, n):
        xr=np.random.rand(n)[:, None]
        x=(self.b-self.a)*xr+self.a
        t = self.t0
        return np.hstack((x, np.full([len(x), 1], t)))
    def random_final_points(self, n):
        xr=np.random.rand(n)[:, None]
        x=(self.b-self.a)*xr+self.a
        t = self.t1
        return np.hstack((x, np.full([len(x), 1], t)))
