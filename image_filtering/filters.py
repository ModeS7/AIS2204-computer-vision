import cv2 as cv
import numpy as np
import scipy.signal as sig

frame =cv.imread( "lenna.ppm" )
grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

cv.imshow("img",grey)


(n,m) = grey.shape
new = np.zeros((n-6,m-6),dtype=np.uint8)

for i in range(n-6):
   for j in range(m-6):
      orig = grey[i:i+7,j:j+7]
      new[i,j] = round(sum(orig.flatten())/49)

#cv.imshow("img1",new)
#cv.waitKey(0)

f7 = np.ones((7,7)) /  49

#print(f7)

g7 = sig.convolve2d(grey,f7).astype(np.uint8)
c7 = cv.filter2D(grey,-1,f7)

#cv.imshow("filtered",c7)
#cv.waitKey(0)

circle = np.array([[0,0,1,1,1,0,0],
   [0,1,1,1,1,1,0],
   [1,1,1,1,1,1,1],
   [1,1,1,1,1,1,1],
   [1,1,1,1,1,1,1],
   [0,1,1,1,1,1,0],
   [0,0,1,1,1,0,0]])/37

cir = sig.convolve2d(grey,circle).astype(np.uint8)

#cv.imshow("filtered1",cir)
#cv.waitKey(0)


def gauss(x, y, sigma=1):
   c1 = 1 / (2 * np.pi * sigma ** 2)
   c2 = 2 * sigma ** 2
   return c1 * np.exp(-(x ** 2 + y ** 2) / c2)

t = 3
B = [ [ gauss(x,y) for x in range(-t,t+1) ] for y in range(-t,t+1) ]

A = np.array(B)

gaussim = sig.convolve2d(grey,A).astype(np.uint8)

#cv.imshow("gauss",gaussim)
#cv.waitKey(0)

gaussaprox = np.array([[1,2,1],[2,4,2],[1,2,1]])/16

gaussimaprox = sig.convolve2d(grey,gaussaprox).astype(np.uint8)

#cv.imshow("gaussaprox",gaussimaprox)
#cv.waitKey(0)


def gnoise(img, sigma=1):
   (m, n) = grey.shape
   noise = np.random.randn(m, n) * sigma
   return (grey + noise).astype(np.uint8)


def snoise(img, p=0.05):
   (m, n) = grey.shape
   noise = (np.random.rand(m, n) > (1 - p)).astype(np.uint8) * 255
   return np.maximum(grey, noise)


def pnoise(img, p=0.1):
   (m, n) = grey.shape
   noise = (np.random.rand(m, n) > p).astype(np.uint8) * 255
   return np.minimum(grey, noise)

#cv.imshow("noise",pnoise(frame))
#cv.waitKey(0)


filter12 = np.array([[0,1,0],[1,-4,1],[0,1,0]])/8
filter1 = filter12 + 0.5
filter1im = sig.convolve2d(grey,filter1).astype(np.uint8)

filter22 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8
filter2 = filter22 + 0.5
filter2im = sig.convolve2d(grey,filter2).astype(np.uint8)

filter32 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8
filter3 = filter32 + 0.5
filter3im = sig.convolve2d(grey,filter3).astype(np.uint8)


cv.imshow("filter1",filter1im)
cv.imshow("filter2",filter2im)
cv.imshow("filter3",filter3im)
cv.waitKey(0)