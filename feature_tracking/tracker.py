import cv2 as cv
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import linalg as LA


img0 =cv.imread( "1.jpg" )
img1 =cv.imread( "2.jpg" )
img2 =cv.imread( "3.jpg" )
img3 =cv.imread( "4.jpg" )
img4 =cv.imread( "5.jpg" )

dst = []
cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	) ->	dst


def img_scale_grey(img, scl):
    scale_percent = scl  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img0 = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img0, cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

def spatial_derivative(grey):



    #kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
    #ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8

    Ix = sig.convolve2d(grey, kx, mode='same').astype(np.uint8)
    Iy = sig.convolve2d(grey, ky, mode='same').astype(np.uint8)

    #imfxc = cv.filter2D(grey, -1, fx)  #here to remind me of this function
    #imfyc = cv.filter2D(grey, -1, fy)

    return Ix, Iy

def H_corner_detector(img, grey, bsize, ksize, k, T):

    #bsize = 2
    #ksize = 11
    #k = 0.08
    # T = 0.4 # Threshold c_image = img

    c_x = cv.cornerHarris(grey, bsize, ksize, k)

    xmax = 0; ymax = 0; (ymin, xmin, un) = img.shape
    x = img.shape[0]
    y = img.shape[1]
    corner = np.array(np.zeros((x, y), dtype=np.uint8), ndmin=2)
    for i in range(x):
        for j in range(y):
            if c_x[i, j] > T:
                cv.circle(img, (j, i), 1, (0, 0, 255), 1)

                corner[i, j] = 1

                if j > xmax:
                    xmax = j
                if j < xmin:
                    xmin = j
                if i > ymax:
                    ymax = i
                if i < ymin:
                    ymin = i

    cv.rectangle(img, (xmax+10, ymax+10), (xmin-10, ymin-10), (0,0,255), 2)
    return corner

def corner_function(corner, greyt, greyt1):
    x = corner.shape[0]
    y = corner.shape[1]
    G = np.array(np.zeros((x, y, 2, 2),dtype=np.uint8),ndmin=4)
    b = np.array(np.zeros((x, y, 2, 1),dtype=np.uint8),ndmin=4)
    img0dt = np.array(np.zeros((x, y), dtype=np.uint8), ndmin=2)
    G_ = np.array(np.zeros((x, y, 2, 2), dtype=np.uint8), ndmin=4)
    u = np.array(np.zeros((x, y), dtype=np.uint8), ndmin=2)
    imfx, imfy = spatial_derivative(greyt)
    for i in range(x):
        for j in range(y):
            if corner[i, j] == 1:
                img0dt[i, j] = greyt1[i, j] - greyt[i, j]
#                G[i, j, 0, 0] = imfx[i,j] ** 2
 #               G[i, j, 0, 1] = imfx[i,j] * imfy[i,j]
  #              G[i, j, 1, 0] = imfx[i,j] * imfy[i,j]
   #             G[i, j, 1, 1] = imfy[i,j] ** 2

#                b[i, j, 0, 0] = imfx[i,j] * img0dt[i,j]
 #               b[i, j, 1, 0] = imfy[i,j] * img0dt[i,j]
  #              print(G[i, j])
   #             G_[i, j] = - np.linalg.inv(G[i, j])
    #            u[i, j] = np.linalg.solve(G[i, j], b[i, j])

    return

#def RemoveClusters

#def PerTileTuning




#combi = cv.hconcat([imfxc/2, imfyc/2])
#combi1 = cv.hconcat([imfx, imfy])

#cv.imshow("img",combi)
#cv.waitKey(0)


#(nx,mx) = imfx.shape
#(ny,my) = imfy.shape


#G = np.array(np.zeros((nx, mx, 2, 2),dtype=np.uint8),ndmin=4)
#Geig = np.array(np.zeros((nx, mx, 2),dtype=np.uint8),ndmin=3)
#Geigmax = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
#Geigmin = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
#Geigsum = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)
#Geigpro = np.array(np.zeros((nx, mx),dtype=np.uint8),ndmin=2)


#for i in range(nx):
 #   for j in range(mx):
  #      G[i, j, 0, 0] = imfx[i,j] ** 2
   #     G[i, j, 0, 1] = imfx[i,j] * imfy[i,j]
    #    G[i, j, 1, 0] = imfx[i,j] * imfy[i,j]
     #   G[i, j, 1, 1] = imfy[i,j] ** 2

      #  Geig[i, j] = LA.eigvals(G[i, j])
       # Geigmax[i, j] = np.amax(Geig[i, j])
        #Geigmin[i, j] = np.amin(Geig[i, j])
        #Geigsum[i, j] = Geig[i, j, 0] + Geig[i, j, 1]
        #Geigpro[i, j] = Geig[i, j, 0] * Geig[i, j, 1]

#combi2 = cv.hconcat([imf2, imf3, Geigmax])
#combi3 = cv.hconcat([Geigmin, Geigsum, Geigpro])
#combi23 = cv.vconcat([combi2, combi3])

#cv.imshow("img1",imfx)
#cv.imshow("img2",Geigmax)
#cv.imshow("img3",Geigmin)
#cv.imshow("img4",Geigsum)
#cv.imshow("img5",Geigpro)


scale = 30
frame0, grey0 = img_scale_grey(img0, scale)
frame1, grey1 = img_scale_grey(img1, scale)

#2, 11, 0.04, 1
bsize, ksize, k, T = 2, 7, 0.0, 0.004
corner0 = H_corner_detector(frame0, grey0, bsize, ksize, k, T)

corner_function(corner0, grey0, grey1)



#combo0 = cv.hconcat([img0dt, img0IxIdt, img0IyIdt])
#combo1 = cv.hconcat([img0Ix2, img0IxIy, img0Iy2])
#combo01 = cv.vconcat([combo0, combo1])
#cv.imshow("img",combo01)
#cv.imshow("img",img0dt)
cv.imshow("img",frame0)
cv.waitKey(0)

