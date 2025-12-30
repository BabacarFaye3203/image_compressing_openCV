import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img=cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(400,400))


def EQM(M1,M2):
    N,M=M1.shape
    eqm=0
    for x in range(N):
        for y in range(M):
            eqm=eqm+(M1[x,y]-M2[x,y])**2
    eqm=eqm/(N*M)
    return eqm
    
def PSNR(M1,M2):
    d=255
    eq=EQM(M1,M2)
    pnsr=10*math.log10(d**2/eq)
    return pnsr

def Coeff_Haar(img):
    n,m=img.shape
    A=np.zeros([m,n],dtype=np.float64)
    for i in range(n):
        for j in range(m):
            if j<=n/2 and i==2*j+1 or i==2*j:
                A[i, j]=1/2
            else:
                if j>n/2 and j<=n and i==2*(j-n/2)-1:
                     A[i, j]=1/2
                else:
                    if j>n/2 and j<=n and i==2*(j-n/2):
                        A[i,j]=-1/2
                    else:
                        if i>n:
                            A[i, j]=1
                        else:
                            A[i, j]=0
    return A


def compression_haar(M, A, niveau):
    sc = M.copy()
    for k in range(niveau):
        sc = np.dot(A.T, np.dot(sc, A))
    return sc

A=Coeff_Haar(img)

img_compr = compression_haar(img, A, 1)
psnr_compressee = PSNR(img, img_compr)
print(psnr_compressee)


plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Originale")
plt.subplot(1, 2, 2)
plt.imshow(img_compr, cmap='gray')
plt.title("Compressée")
plt.show()