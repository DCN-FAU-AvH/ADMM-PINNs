import numpy as np
import scipy as scp
from mesh import rectangleMesh, quadpts,TriMesh2D, quadpts1
class Square_mesh:
    def __init__(self,n=64,x_range=(0,1),y_range=(0,1)):
        self.node, self.elem=rectangleMesh(x_range, y_range, 1/n)
        T = TriMesh2D(self.node,self.elem)
        T.update_auxstructure()
        T.update_gradbasis()
        self.isBdNode = T.isBdNode
        self.Dphi = T.Dlambda
        self.area = T.area
        self.phi, self.weight = quadpts()
        self.nQuad = len(self.phi)
        self.nDoF = len(self.node)
        self.allEdge = np.r_[self.elem[:,[1,2]], self.elem[:,[2,0]], self.elem[:,[0,1]]]
        self.allEdge = np.sort(self.allEdge, axis=1)
        self.edge, E2e, e2E, counts = np.unique(self.allEdge, 
                            return_index=True, 
                            return_inverse=True, 
                            return_counts=True,
                            axis=0)
        elem2edge = e2E.reshape(3,-1).T
        self.isBdEdge = (counts==1)
#     def Stiffness_Matrix_backup(self, diffusion):
#         pxy=(self.node[self.elem[:,0]]+self.node[self.elem[:,1]]+self.node[self.elem[:,2]])/3
#         alpha11, alpha12, alpha22=diffusion(pxy)
#         A = scp.sparse.csc_matrix((self.nDoF,self.nDoF))
#         for i in range(3):
#             for j in range(3):
#                 Aij_11 = self.area*alpha11*(self.Dphi[:,0,i]*self.Dphi[:,0,j])
#                 Aij_12 = self.area*alpha12*(self.Dphi[:,0,i]*self.Dphi[:,1,j])
#                 Aij_21 = self.area*alpha12*(self.Dphi[:,1,i]*self.Dphi[:,0,j])
#                 Aij_22 = self.area*alpha22*(self.Dphi[:,1,i]*self.Dphi[:,1,j])
#                 Aij =Aij_11+Aij_12+Aij_21+Aij_22
#                 A += scp.sparse.csc_matrix((Aij, (self.elem[:,i],self.elem[:,j])), shape=(self.nDoF,self.nDoF))  
#         return A
    def Stiffness_Matrix(self):
        A = scp.sparse.csc_matrix((self.nDoF,self.nDoF))
        for i in range(3):
            for j in range(3):
                Aij = self.area*(self.Dphi[...,i]*self.Dphi[...,j]).sum(axis=-1)
                A += scp.sparse.csc_matrix((Aij, (self.elem[:,i],self.elem[:,j])), shape=(self.nDoF,self.nDoF))  
        return A
    def Mass_Matrix(self):       
        M = scp.sparse.csc_matrix((self.nDoF, self.nDoF))
        for i in range(3):
            for j in range(3):
        # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$ 
                Mij = (1+(i==j))/12*self.area
                M += scp.sparse.csc_matrix((Mij, (self.elem[:,i],self.elem[:,j])), shape=(self.nDoF,self.nDoF))
        return M
    def aux_area(self):
        area_vec=np.zeros([len(self.area),1])
        area_vec[:,0]=self.area/3
        area_aux=np.hstack((area_vec,area_vec,area_vec))
        aux=np.zeros([self.nDoF,1])
        aux[:,0]=np.bincount(self.elem.ravel(), weights=area_aux.ravel())/np.sum(self.area)
        return aux
    def righthand_Neumann(self,j_Neumann):
        Neumann=self.edge[self.isBdEdge]
        Neumann_vector=np.hstack((Neumann[:,0],Neumann[:,1]))
        Neumann_node,ic=np.unique(Neumann_vector, return_inverse=True)
        n_Neumann_vector=len(Neumann_vector)
        Pro=scp.sparse.csr_matrix((np.ones(n_Neumann_vector), (range(n_Neumann_vector),Neumann_node[ic])), shape=(n_Neumann_vector,self.nDoF))
        el=np.zeros([len(Neumann),1])
        el[:,0]=np.sqrt((self.node[Neumann[:,0],0]-self.node[Neumann[:,1],0])**2+(self.node[Neumann[:,0],1]-self.node[Neumann[:,1],1])**2)
        [lambdagN,weightgN] = quadpts1(1)
        phigN = lambdagN;                
        nQuadgN = len(lambdagN);
        ge = np.zeros([len(Neumann),2]);
        for pp in range(nQuadgN):
            ppxy=lambdagN[pp,0]*self.node[Neumann[:,0],:]+lambdagN[pp,1]*self.node[Neumann[:,1],:]
            gNp=j_Neumann(ppxy)[:,0]
            for igN in range(2):
                ge[:,igN]=ge[:,igN]+weightgN[pp]*phigN[pp,igN]*gNp
        ge=np.vstack((ge[:,0:1]*el,ge[:,1:]*el))
        b_Neumann=np.zeros([self.nDoF,1])
        b_Neumann=Pro.T.dot(ge)
        return b_Neumann
#     def righthand_vector(self, f_fun):
#         pxy=(self.node[self.elem[:,0]]+self.node[self.elem[:,1]]+self.node[self.elem[:,2]])/3
#         x1=pxy[:,0]
#         x2=pxy[:,1]
#         return f_fun(pxy)
#         self.A=A
#         self.M=M
#         f_right= 2*np.float32((abs(x1)**2+abs(x2)**2<=0.25))-np.pi/8
#         self.b_exact=M.dot(f_right)
#         part1=1.*(0<x1)*(x1<=1)*(x2==-1)-1.*(-1<=x1)*(x1<=0)*(x2==1)
#         part2=2.*(0<x1)*(x1<=1)*(x2==1)-2.*(-1<=x1)*(x1<=0)*(x2==-1)
#         part3=3.*(x1==-1)*(-1<x2)*(x2<=0)-3.*(x1==1)*(0<x2)*(x2<1)
#         part4=4.*(x1==1)*(-1<x2)*(x2<=0)-4.*(x1==-1)*(0<x2)*(x2<1)
#         self.j_Neumann=part1+part2+part3+part4
#         area_vec=np.zeros([len(area),1])
#         area_vec[:,0]=area/3
#         area_aux=np.hstack((area_vec,area_vec,area_vec))
#         aux=np.zeros([nDoF,1])
#         aux[:,0]=np.bincount(elem.ravel(), weights=area_aux.ravel())/np.sum(area)
# def OMEGA11(pxy):
# #     pxy=(node[elem[:,0]]+node[elem[:,1]]+node[elem[:,2]])/3
#     x1=pxy[:,0]
#     x2=pxy[:,1]
#     return np.float32((abs(x1)<=0.5)*(abs(x2)<0.5))
# def OMEGA22(pxy):
# #     pxy=(node[elem[:,0]]+node[elem[:,1]]+node[elem[:,2]])/3
#     x1=pxy[:,0]
#     x2=pxy[:,1]
#     return np.float32((abs(x1)**2+abs(x2)**2<=0.25))
# def OMEGA12(pxy):
# #     pxy=(node[elem[:,0]]+node[elem[:,1]]+node[elem[:,2]])/3
#     x1=pxy[:,0]
#     x2=pxy[:,1]
#     return np.float32((abs(x1)+abs(x2)<0.5))
# def f_righthand(pxy):
#     x1=pxy[:,0]
#     x2=pxy[:,1]
#     return 2*np.float32((abs(x1)**2+abs(x2)**2<=0.25))-np.pi/8
# #     return 2*np.sin(np.pi*x1)*np.sin(np.pi*x2)
# def j_Neumann(pxy):
#     x1=pxy[:,0]
#     x2=pxy[:,1]
#     part1=1.*(0<x1)*(x1<=1)*(x2==-1)-1.*(-1<=x1)*(x1<=0)*(x2==1)
#     part2=2.*(0<x1)*(x1<=1)*(x2==1)-2.*(-1<=x1)*(x1<=0)*(x2==-1)
#     part3=3.*(x1==-1)*(-1<x2)*(x2<=0)-3.*(x1==1)*(0<x2)*(x2<1)
#     part4=4.*(x1==1)*(-1<x2)*(x2<=0)-4.*(x1==-1)*(0<x2)*(x2<1)
#     return part1+part2+part3+part4