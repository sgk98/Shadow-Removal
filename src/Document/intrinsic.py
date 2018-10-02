import cv2
import numpy as np


def get_shading(rim,mask):
	im=rim
	

	im=cv2.copyMakeBorder(rim, top=100, bottom=100, left=100, right=100, borderType= cv2.BORDER_CONSTANT, value=255)
	mask=cv2.copyMakeBorder(mask, top=100, bottom=100, left=100, right=100, borderType= cv2.BORDER_CONSTANT, value=255)
	mask=mask/255
	shading=np.zeros(im.shape)
	for i in range(len(im)):
		print(i)
		for j in range(len(im[i])):
			if mask[i][j]>0:
				shading[i][j]=im[i][j]
				#print("HELLO")
			else:
				dx=25
				dy=25
				while np.count_nonzero(mask[i-dx:i+dx,j-dy:j+dy])<25:
					dx=dx*2
					dy=dy*2
					#print(dx,dy,i,j)
				curr=0.0
				tot=0.0
				for it1 in range(i-dx,i+dx):
					for it2 in range(j-dy,j+dy):
						if mask[it1][it2]>0:
							curr+=im[it1][it2]
						tot+=mask[it1][it2]
				shading[i][j]=curr/tot



	return shading[100:-100,100:-100]



def get_reflectance(im,shading):
	reflectance=np.zeros(im.shape)
	for i in range(len(im)):
		for j in range(len(im[i])):
	
			reflectance[i][j]=255*((im[i][j]*1.0)/shading[i][j])
	#reflectance=reflectance[30:-30,30:-30]

	return reflectance

if __name__=="__main__":



	rim=cv2.imread('doc1.png',0)
	im1=cv2.imread('bin1.png',0)
	rim=cv2.resize(rim,None,fx=0.3,fy=0.3)
	shading=get_shading(rim,im1)
	reflectance=get_reflectance(rim,shading)
	cv2.imwrite('shading.png',shading)
	cv2.imwrite('reflectance.png',reflectance)



