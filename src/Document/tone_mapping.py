import numpy as np
import cv2


shading=cv2.imread('shading.png',0)
reflectance=cv2.imread('reflectance.png',0)
M=cv2.imread('bin1.png',0)
M=M/255
im1=cv2.imread('doc1.png',0)

im1=cv2.resize(im1,None,fx=0.3,fy=0.3)



ret,shadow_mask = cv2.threshold(shading,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

gm=0
num=0.0
dem=0.0
for i in range(len(im1)):
	for j in range(len(im1[i])):
		num+=im1[i][j]*shadow_mask[i][j]*M[i][j]
		dem+=M[i][j]*shadow_mask[i][j]

gm=num/dem
print(gm)
reflectance=reflectance*gm

cv2.imwrite('shadow_mask.png',shadow_mask)
cv2.imwrite('final.png',reflectance)