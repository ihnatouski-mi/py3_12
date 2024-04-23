"""
    Least squares by Gauss     
"""
import numpy as np
from numpy.typing import NDArray 
import matplotlib.pyplot as plt

#=============================================================================
def fnGetData()->tuple:
    X=np.array([650,550,480,580,690,870,840,750,920,1050])
    Y=np.array([12.6,13.5,14.3,13.2,12.3,11.5,11.6,12.0,11.3,11.0])
    return (X,Y)
#=============================================================================
def fnGetLeastSquares(X,Y)->NDArray:
    n=len(X)
    x=sum(X)
    y=sum(Y)
    xy=sum(X*Y)
    x2=sum(X*X)
    M=np.zeros((2,3))
    M[0,0]=x2; M[0,1]=x; M[0,2]=xy
    M[1,0]=x;  M[1,1]=n; M[1,2]=y
    return M 
#=============================================================================
def fnGauss(A)->NDArray:
    (lines,columns)=A.shape
    sol=np.zeros(lines)
    column0=0
    for j0 in range(0,lines-1):                 # от первой (№ 0) до предпоследней
        #print(f"j0={j0}")
        for j in range(j0+1,lines):             # от следующей за j0 до последней 
            #print(f"j={j}")
            s=A[j,column0]/A[j0,column0]
            for i in range(column0,columns):    #  вдоль укорачивающейся слева строки
                #print(f"i={i}")
                A[j,i]-=A[j0,i]*s
        column0+=1
    for j in range(lines-1,0-1,-1):             #  от последней до первой (№ 0) 
        #print(f"j={j}") 
        s=0
        for i in range(columns-2,j-1,-1):       #  с предпоследней позиции вдоль строки в обратном порядке
            #print(f"i={i}")
            s+=A[j,i]*sol[i]
        sol[j]=(A[j,columns-1]-s)/A[j,j]
    return sol
#=============================================================================
def fnGetLine(X,K)->NDArray:
    n=len(X)
    Y=np.zeros(n)
    Y=K[0]*X+K[1]
    return Y
#=============================================================================
def fnPlot(X,Y0,Y1,sTitle)->None:
    plt.figure()
    plt.title(sTitle)
    plt.scatter(X,Y0,linewidths=4,color='blue')
    plt.plot(X,Y1,linewidth=3,color='red')
    plt.show()
#=============================================================================
def fnTestMatrixVector()->None:
    """ Linear equation solution by inversion of matrix 
        and  multiplication by vector"""
    A5=np.array([[3.0,  5.0, -2.0,  3.0,-18.0],
                 [11.0,-3.0, 32.0, -2.0, -1.0],
                 [6.0, -3.0,  8.0,-21.0,  1.0],
                 [1.0, -3.0,  2.0, -2.0, -1.0],
                 [6.0,  7.0, -3.0, 10.0,-12.0]])
    B=np.array([1.0, 2.0,-1.0,4.0,-4.0])
    A_5=np.linalg.inv(A5)
    sol=np.dot(A_5,B)
    print("Linear equation solution by inversion of matrix and  multiplication by vector:")
    print(sol)
#=============================================================================
def fnTestGauss()->None:
    """ Linear equation solution by Gauss """
    A5=np.array([[3.0,  5.0, -2.0,  3.0,-18.0,  1.0],
                 [11.0,-3.0, 32.0, -2.0, -1.0,  2.0],
                 [6.0, -3.0,  8.0,-21.0,  1.0, -1.0],
                 [1.0, -3.0,  2.0, -2.0, -1.0,  4.0],
                 [6.0,  7.0, -3.0, 10.0,-12.0, -4.0]])
    sol=fnGauss(A5)
    print("Linear equation solution by Gauss:")
    print(sol)
#=============================================================================
def fnMain()->None:
    print(__name__)
    fnTestMatrixVector()
    fnTestGauss()
    (X0,Y0)=fnGetData()
    M=fnGetLeastSquares(X0,Y0)
    K=fnGauss(M)
    Y1=fnGetLine(X0,K)
    fnPlot(X0,Y0,Y1,f"y={K[0]:.{8}}*x+{K[1]:.{7}}")
#=============================================================================
if __name__=='__main__':
    fnMain()
