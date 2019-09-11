import numpy as np
from sympy import *
from sympy.vector import CoordSys3D, gradient
import os
import scipy.io
from scipy.io import loadmat
import cvxpy as cvx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import spams


def soft(x, T):
    return np.multiply(np.sign(x), np.fmax(abs(x) - T, 0))


def update_r(M, B, x, a, u, v1, N):
    T = 1/N
    w = x - M * (B * a + u) - v1 / N
    r = soft(w, T)
    return r


def update_b(O, v2, a, N):
    T = O / N
    w = v2 / N + a
    b = soft(w, T)
    return b


class update_a:
    def __init__(self, B, M, r, x, u, v1, b, v2, N):
        self.B = np.mat(B)
        self.M = np.mat(M)
        self.r = np.mat(r)
        self.x = np.mat(x)
        self.u = np.mat(u)
        self.v1 = np.mat(v1)
        self.b = np.mat(b)
        self.v2 = np.mat(v2)
        A = np.transpose(self.B) * np.transpose(self.M) * self.M * self.B
        A = A + np.eye(A.shape[0], dtype=float)
        a = 2 * (np.transpose(self.r - self.x + self.M * self.u + self.v1 / N) * self.M * self.B - self.b.T + self.v2.T / N)
        w1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        w2 = np.hstack((a, np.zeros((a.shape[0], 1))))
        w = np.vstack((w1, w2))
        self.w = np.mat(w)


    def L2(self, P, Q, G, deta):
        L = cvx.trace(self.w * Q) + cvx.trace(np.transpose(G) * (Q - P)) + (deta / 2) * cvx.norm(Q - P, p='fro', axis=None) ** 2
        return L


    def update_Q(self, P, G, deta, q, m, C, L):
        Q = cvx.Variable((q,q))
        constraints = []
        for i in range(m):
            Omiga = np.vstack((np.hstack((np.transpose(self.B) * np.transpose(C[i]) * C[i] * self.B,
                      np.transpose(self.B) * np.transpose(C[i]) * C[i] * self.u)),
                     np.hstack((np.transpose(self.u) * np.transpose(C[i]) * C[i] * self.B,
                      np.transpose(self.u) * np.transpose(C[i]) * C[i] * self.u - L[i]))))
            constraints += [cvx.trace(Omiga * Q) == 0]
        constraints += [Q[q-1,q-1]==1]
        L2 = self.L2(P, Q, G, deta)
        obj = cvx.Minimize(L2)
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.SCS, verbose=False, eps=1e-4, alpha=1.8, max_iters=100000, scale=5, use_indirect=True)
        print(prob.status)
        return Q.value

    @staticmethod
    def update_P(Q, G, deta):
        Q1 = Q + (2 / deta) * G
        S = (Q1 + np.transpose(Q1)) / 2
        a, b = np.linalg.eig(S)
        b = b.T
        v = max(a)
        k = list(a).index(v)
        v1 = np.mat(b[k]).T
        P = max(v, 0) * v1 * np.transpose(v1)
        return P

    def get_Q(self, P, G, deta, q, m, C, L):
        k = 0
        a=[]
        b=[]
        Q=np.zeros((q,q))
        P=Q
        while True:
            Q1=Q
            P1=P
            Q = self.update_Q(P, G, deta, q, m, C, L)
            P = self.update_P(Q, G, deta) 
            G = G + deta * (Q - P)
            deta = min(deta*1.1, 40000000000)
            print(np.linalg.norm(P - Q, ord = 2),np.linalg.norm(Q-Q1,ord=2),np.linalg.norm(P-P1,ord=2))
            a.append(np.trace(self.w*Q))
            b.append(np.linalg.norm(P - Q, ord = 2))
            k = k + 1
            if max(np.linalg.norm(P - Q, ord = 2),np.linalg.norm(Q-Q1,ord=2),np.linalg.norm(P-P1,ord=2))<0.03:
            	return Q

    @staticmethod
    def get_a(Q,q):
    	r = Q.shape[0]
    	a = Q[r-1]
    	a = np.mat(a[:-1]).T
    	a = a*(1/Q[q-1,q-1])
    	return a


def Pose_Estimation(M, B, x, u, O, m):
	'''the square of length of segments'''
    L = [0.494, 0.658, 1.055, 1, 0.494, 0.658, 1.055, 1, 0.554, 0.554, 1.357, 1.357, 0.5]
    for i in range(13):
        L[i] = L[i] * ((u[35]-u[32])**2 + (u[36]-u[33])**2 + (u[34]-u[31])**2)
    E = []
    C = []
    for j in range(14):
        e = np.zeros([3, 42])
        e[0][3*j] = 1
        e[1][3*j+1] = 1
        e[2][3*j+2] = 1
        E.append(e)
    for i in range(4):
        c1 = E[3*i] - E[3*i+1]
        c2 = E[3*i+1] - E[3*i+2]
        C.append(c1)
        C.append(c2)
    C.append(E[2]-E[12])
    C.append(E[8]-E[12])
    C.append(E[3]-E[12])
    C.append(E[9]-E[12])
    C.append(E[12]-E[13])
    a = spams.lasso(x - M*u,np.asfortranarray(M*B),return_reg_path = False, lambda1=0.01)
    a = a.toarray()
    idx = []
    for i in range(len(a)):
       	if a[i,0] != 0:
       		idx.append(i)
    a = np.mat([a[i,0] for i in idx]).T
    B = np.mat([np.matrix.tolist(B.T)[i] for i in idx]).T
    q = len(a)+1
    G = 0.000001*np.random.rand(q,q)
    v1 = 0*np.random.rand(28, 1)
    v2 = 0*np.random.rand(q-1, 1)
    N = 100
    deta = 20
    A=[0]
    k=0
    while True:
        a1 = a
        r = update_r(M, B, x, a, u, v1, N)
        b = update_b(O, v2, a, N)
        z = np.hstack((b.T,np.mat([1]))).T
        P = z*z.T
        update = update_a(B, M, r, x, u, v1, b, v2, N)
        Q = update.get_Q(P, G, deta, q, m, C, L)
        a = update.get_a(Q,q)
        v1 = v1 + N * (r - x + M * (B * a + u))
        v2 = v2 + N * (a - b)  
        A.append(np.linalg.norm(a-a1,ord=2)+np.linalg.norm(r, ord=1)+0.01*np.linalg.norm(b,ord=1))
        k+=1
        if k==20 or abs(A[-1]-A[-2])<0.001:
        	break
    return a, B


def update_R(m1, m2, X, H, Y, tao): 
    T = 1/tao
    M = np.vstack((np.transpose(m1), np.transpose(m2)))
    w = X - H/tao - M * Y
    R = soft(w, T)
    return R


def update_m1(m1, m2, R, X, H, Y, xi, tao):
    x,y,z=symbols('x y z', real=True)
    m10 = np.mat([x, y, z]).T
    F0 = (Matrix(np.vstack((np.transpose(m10), np.transpose(m2))) * Y + R - X + H/tao).norm(ord='fro'))**2 + Matrix((m10.T * m2 + xi/tao)**2).norm()
    m1=[]
    m10=solve([diff(F0,x),diff(F0,y),diff(F0,z)],[x,y,z])
    m1.append(float(m10[x]))
    m1.append(float(m10[y]))
    m1.append(float(m10[z]))
    print(m1)
    return np.mat(m1).T


def update_m2(m1, m2, R, X, H, Y, xi, tao):
    x,y,z=symbols('x y z', real=True)
    m20 = np.mat([x, y, z]).T
    F0 = (Matrix(np.vstack((np.transpose(m1), np.transpose(m20))) * Y + R - X + H/tao).norm(ord='fro'))**2 + Matrix((np.transpose(m1) * m20 + xi/tao)**2).norm()
    m2=[]
    m20=solve([diff(F0,x),diff(F0,y),diff(F0,z)],[x,y,z])
    m2.append(float(m20[x]))
    m2.append(float(m20[y]))
    m2.append(float(m20[z]))
    print(m2)
    return np.mat(m2).T


def Camera_Parameter_Estimation(X, Y, m1, m2):
    H = np.random.rand(2, 14)
    xi = 0.3
    tao = 20
    k=0
    T=[]
    W=[]
    R=H
    while True:
        m = m1
        m3 = m2
        R1=R
        R = update_R(m1, m2, X, H, Y, tao)
        m1 = update_m1(m1, m2, R, X, H, Y, xi, tao)
        m2 = update_m2(m1, m2, R, X, H, Y, xi, tao)
        H = H + tao * (np.vstack((np.transpose(m1), np.transpose(m2))) * Y + R - X)
        xi = xi + tao * np.transpose(m1) * m2
        k=k+1
        W.append(np.linalg.norm(R, ord=1)-np.linalg.norm(R1, ord=1))
        if abs(W[-1]+np.linalg.norm(m1 - m, ord = 2) + np.linalg.norm(m2 - m3, ord = 2)) < 0.05:
        	break
        else:
            continue
    return m1, m2


if __name__ =='__main__':
    image_file = '' # load image file here 

    ''' 
    get the set of pose in CMU Mocap dataset:

    B = amc_parser.test_all()
    if not os.path.exists('./saved_model'):
    	os.makedirs('./saved_model')
    scipy.io.savemat('./saved_model/B.mat', mdict={'poses':B})

    '''
    # load poses
    B = loadmat("B.mat")
    B = B['poses']

    # detect 2D pose
    X = pose_detect.pose_detect(image_file)
    X = np.mat(X,dtype=float).T

   # X = np.mat([[753, 745, 761, 794, 834, 802, 894, 870, 850, 850, 890, 854, 810, 826],
   #		    [668, 587, 494, 656, 778, 903, 612, 567, 486, 644, 774, 887, 486, 421]])

   # mean-center and normalization
    a = np.sum(X[0])/14
    b = np.sum(X[1])/14
    for i in range(14):
    	X[0,i]=X[0,i]-a
    	X[1,i]=X[1,i]-b
    X = X /np.sqrt(np.sum(np.multiply(X,X)))

    # initialize M
    m1 = np.mat([ 1.8,  1.85,  -0.1]).T
    m2 = np.mat([ 0.3,  -0.35,  -1]).T

    # rewrite 2D pose as 2n * 1
    x = []
    for i in range(14):
        x.append(X[0,i])
        x.append(X[1,i])
    x = np.mat(x).T

    # get mean 3D pose as 3n * 1
    u = []
    for i in range(42):
        k = 0
        for j in B:
           k = k+j[i]
        u.append(k/len(B))
    u = np.mat(u).T

    # rewrite mean pose as 3 * n
    Y = [[u[3*i], u[3*i+1], u[3*i+2]] for i in range(14)]
    Y = np.mat(Y).T

    I = np.eye(14)
    m = 13
    O = 0.01

    # load bases
    B = loadmat('B_spams_50_1.mat')
    B = np.mat(B['B'])
    
    k=0
    print('X:',X)
    print('Y:',Y)
    print('B:',B)
    while True:
        y = Y
        m1, m2 = Camera_Parameter_Estimation(X, Y, m1, m2)
        M0 = np.vstack((np.transpose(m1), np.transpose(m2)))
        M = np.kron(M0, I)
        a, B1 = Pose_Estimation(M, B, x, u, O, m)
        Y = B1 * a + u
        Y = Y.reshape((14,3)).T
        k=k+1
        if k>=20 or np.linalg.norm(Y-y, ord=2)<0.1:
            break

    # show 3D pose

    fig = plt.figure()
    ax = p3d.Axes3D(fig)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    x1 = Y[0]
    y1 = Y[1]
    z1 = Y[2]
    x=[]
    y=[]
    z=[]
    for i in range(14):
        x.append(x1[0,i])
        y.append(y1[0,i])
        z.append(z1[0,i])
    plt.plot(x, y, z, 'b.')
    x2 = [[x[0],x[1]], [x[1],x[2]], [x[3],x[4]], [x[4],x[5]], [x[6],x[7]], [x[7],x[8]], [x[9],x[10]], [x[10],x[11]], [x[2],x[12]], [x[3],x[12]], [x[9],x[12]], [x[8],x[12]], [x[12],x[13]]]
    y2 = [[y[0],y[1]], [y[1],y[2]], [y[3],y[4]], [y[4],y[5]], [y[6],y[7]], [y[7],y[8]], [y[9],y[10]], [y[10],y[11]], [y[2],y[12]], [y[3],y[12]], [y[9],y[12]], [y[8],y[12]], [y[12],y[13]]]
    z2 = [[z[0],z[1]], [z[1],z[2]], [z[3],z[4]], [z[4],z[5]], [z[6],z[7]], [z[7],z[8]], [z[9],z[10]], [z[10],z[11]], [z[2],z[12]], [z[3],z[12]], [z[9],z[12]], [z[8],z[12]], [z[12],z[13]]]
    for i in range(13):
        plt.plot(x2[i], y2[i], z2[i], 'r')
    plt.show()
