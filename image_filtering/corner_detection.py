import cv2 as cv
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import linalg as LA

frame =cv.imread( "lenna.ppm" )
grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#cv.imshow("img",grey)

(n,m) = grey.shape


row = grey[50,:]   # Row number 50
p = row.size

f1= [0.5, -0,5]
d = np.convolve(row,f1)
d1 = d.size

#row_d = sig.convolve(row, f1)
#p1 = row_d.size

#plt.plot(range(d1),d,color="blue",label="Pixel Row")
#plt.show()


f2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8
imf2 = sig.convolve2d(grey,f2).astype(np.uint8)
imf2c = cv.filter2D(grey,-1,f2)
imf2 = imf2

f3 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8
imf3 = sig.convolve2d(grey,f3).astype(np.uint8)
imf3c = cv.filter2D(grey,-1,f3)
imf3 = imf3

combi = cv.hconcat([imf2c/2, imf3c/2])
combi1 = cv.hconcat([imf2, imf3])

#cv.imshow("img",combi)
#cv.waitKey(0)


(nx,mx) = imf2.shape
(ny,my) = imf3.shape

#Gnm= np.array[[imf2(n,m)**2, imf2(n,m)*imf3(n,m)],
#         [imf2(n,m)*imf3(n,m),imf3(n,m)**2]]
G = np.array(np.zeros((nx, mx, 2, 2),dtype=np.uint8),ndmin=4)
Geig = np.array(np.zeros((nx, mx, 2),dtype=np.uint8),ndmin=3)
Geigmax = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
Geigmin = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
Geigsum = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
Geigpro = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)


#for i in range(nx):
#    for j in range(mx):
#        G[i, j, 0, 0] = imf2[i,j] ** 2
#        G[i, j, 0, 1] = imf2[i,j] * imf3[i,j]
#        G[i, j, 1, 0] = imf2[i,j] * imf3[i,j]
 #       G[i, j, 1, 1] = imf3[i,j] ** 2
#
 #       Geig[i, j] = LA.eigvals(G[i, j])
  #      Geigmax[i, j] = np.amax(Geig[i, j])
   #     Geigmin[i, j] = np.amin(Geig[i, j])
 #       Geigsum[i, j] = Geig[i, j, 0] + Geig[i, j, 1]
  #      Geigpro[i, j] = Geig[i, j, 0] * Geig[i, j, 1]


bsize = 2
ksize = 5
k = 0.06

c_x = cv.cornerHarris(grey, bsize, ksize, k)
T = 0.01 # Threshold c_image = img

for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        if c_x[i, j] > T: cv.circle(frame, (j, i), 2, (0, 0, 255), 2)


#combi2 = cv.hconcat([imf2, imf3, Geigmax])
#combi3 = cv.hconcat([Geigmin, Geigsum, Geigpro])
#combi23 = cv.vconcat([combi2, combi3])
cv.imshow("img",frame)
#v.imshow("img1",imf3)
#cv.imshow("img2",Geigmax)
#cv.imshow("img3",Geigmin)
#cv.imshow("img4",Geigsum)
#cv.imshow("img5",Geigpro)
cv.waitKey(0)