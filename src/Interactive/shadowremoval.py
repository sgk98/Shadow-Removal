from skimage import io
from skimage.transform import resize, rescale
from skimage.filters import gaussian
from scipy.spatial.distance import cosine
from skimage.restoration import denoise_bilateral
import skimage.morphology as morph
from skimage import color
import scipy.spatial
import numpy as np

originalImage = io.imread('shadowimage.jpg')[:,:,:3]

imx, imy, _ = originalImage.shape

startPoint = (100,120)

import queue
Q = queue.Queue(1024)
Q.put((startPoint,0))

useImage = np.uint8(255 * denoise_bilateral(originalImage,sigma_color=.06,
                                  sigma_spatial=8,multichannel=True))

neighbours = [(1,0),(-1,0),(0,1),(0,-1)]
iterations = 100
finMask = np.zeros(useImage.shape[0:2])
colList = np.array(useImage[startPoint])[np.newaxis,:]

while not Q.empty():
  cur, lev = Q.get()
  for i in neighbours:
    x,y = (cur[0] + i[0], cur[1] + i[1])
    if x<0 or x>=imx or y<0 or y>=imy or finMask[x,y] != 0:
      continue
    
    score = useImage[cur] - useImage[x,y]
    score = np.sum(score ** 2)
    
    if score < 4:
      finMask[x,y] = 1
      if lev < iterations:
        Q.put(((x,y), lev+1))
        curCol = (useImage[x,y])[np.newaxis,:]
        colList = np.concatenate((colList, curCol), axis=0)
    else:
      finMask[x,y] = 2
      
finMask[finMask==2] = 0
median = np.median(colList, axis=0)

disk = morph.disk(1)
finMaskEroded = morph.binary_closing(finMask, disk)

median = np.zeros((3))
for i in range(3):
  median[i] = np.median(useImage[:,:,i][finMaskEroded==1])

newImage = np.array(useImage)
for i in range(3):
  newImage[:,:,i] = finMaskEroded * originalImage[:,:,i]

cosineImage = np.zeros(originalImage.shape[0:2])
filteredImage = denoise_bilateral(originalImage,sigma_color=0.04,
                                  sigma_spatial=2,multichannel=True)

for i in range(imx):
  for j in range(imy):
    cosineImage[i,j] = np.abs(cosine(originalImage[i,j],median))

oldCosineImage = np.array(cosineImage)

thresh = 2.54409105e-02
cosineImage[cosineImage < thresh] = 1
cosineImage[cosineImage != 1] = 0

msMask = np.array(finMaskEroded).astype(int)
prev_msMask = np.array(msMask)

filteredImage = denoise_bilateral(originalImage,sigma_color=0.04,
                                  sigma_spatial=2,multichannel=True)
ybrImage = color.rgb2ycbcr(filteredImage)

cb = ybrImage[:,:,1]
cr = ybrImage[:,:,2]

def similar_colours(a,b):
  
  mncb = (cb[:,:])[msMask==1].mean()
  mncr = (cr[:,:])[msMask==1].mean()

  #if ybrImage[:,:,0][a]-mn < thresh:
  #print(cb[:,:][a]-cb[:,:][b])
  #print(np.abs(cb[:,:][a]-cb[:,:][b]) + np.abs(cr[:,:][a]-cr[:,:][b]))
  if np.abs(cb[:,:][a]-cb[:,:][b]) + np.abs(cr[:,:][a]-cr[:,:][b]) <= thresh:
    return True
  return False

thresh = 0.001
add = 0
ls = []
ls2 = []

cont = 0

while True:
  
  x,y = np.where(msMask==1)
  ms_sd = ybrImage[x,y,0].std()
  
  x,y = np.where((cosineImage * (1-msMask))==1)
  ml_sd = ybrImage[x,y,0].std()
  
  ls2.append(ml_sd - ms_sd)
  
  if ml_sd <= ms_sd:
    msMask = np.array(prev_msMask)
    break
    
  prev_msMask = np.array(msMask)
  added = 0
  
  xL, yL = np.where(msMask==1)
  for i in range(len(xL)):
    for j in neighbours:
      x,y = xL[i] + j[0], yL[i] + j[1]
      if x<0 or y<0 or x>=imx or y>=imy or msMask[x,y] == 1 or cosineImage[x,y] == 0:
        continue
      if similar_colours((x,y), (xL[i],yL[i])):
        msMask[x,y] = 1
        added += 1
  
  add += added

  if added == 0:
    ls.append(add)    
    add = 0
    print("Added none")
    thresh += 0.005
    io.imshow(msMask)
    plt.show()
    cont += 1
    
  else:
    cont = 0
    
  if cont == 50:
    break

temp = np.array(msMask).astype(int)

msMask = np.array(finMaskEroded).astype(int)
prev_msMask = np.array(msMask)
ybrImage = color.rgb2ycbcr(filteredImage)

def similar_colours(a,b):
  
  mn = (ybrImage[:,:,0])[msMask==1].mean()

  #if ybrImage[:,:,0][a]-mn < thresh:
  if ybrImage[:,:,0][a]-ybrImage[:,:,0][b] < thresh:
    return True
  return False

thresh = 0
add = 0
ls = []
ls2 = []

cont = 0

while True:
  
  # calculating sd of ms, ml
  x,y = np.where(msMask==1)
  ms_sd = ybrImage[x,y,0].std()
  
  x,y = np.where((cosineImage * (1-msMask))==1)
  ml_sd = ybrImage[x,y,0].std()
  
  ls2.append(ml_sd - ms_sd)
  
  #print(ml_sd - ms_sd)
  if ml_sd <= ms_sd:
    msMask = np.array(prev_msMask)
    break
    
  # region growing
  prev_msMask = np.array(msMask)
  added = 0
  
  xL, yL = np.where(msMask==1)
  for i in range(len(xL)):
    for j in neighbours:
      x,y = xL[i] + j[0], yL[i] + j[1]
      if x<0 or y<0 or x>=imx or y>=imy or msMask[x,y] == 1 or cosineImage[x,y] == 0:
        continue
      if similar_colours((x,y), (xL[i],yL[i])):
        msMask[x,y] = 1
        added += 1
  
  add += added

  if added == 0:
    ls.append(add)    
    add = 0
    thresh += 0.2
    cont += 1
    
  else:
    cont = 0
    
  if cont >= 10:
    break

disk = morph.disk(2)
tm = morph.binary_closing(msMask, disk) * morph.binary_closing(temp, disk)

disk = morph.disk(2)
msMask_n = morph.binary_closing(msMask, disk)

mlMask_n = cosineImage * (1-msMask_n)
mlMask_n = morph.binary_opening(mlMask_n, disk)

trimap = np.zeros(cosineImage.shape)
trimap[mlMask_n==1] = 1
trimap[msMask_n==1] = 1

def traverse(x,y,nm):
  Q = queue.Queue(100000)
  Q.put((x,y))
  if nm == 3:
    trimap[x,y] = 3
  
  while not Q.empty():
    x,y = Q.get()
    
    for j in neighbours:
      xn, yn = x + j[0], y + j[1]
      if xn<0 or xn>=imx or yn<0 or yn>=imy:
        continue
      if nm == 3 and mlMask_n[xn,yn] == 1:
        return 0
      if trimap[xn,yn] == 1:
        continue
      if nm == 3:
        if trimap[xn,yn] == 3:
          continue
        Q.put((xn,yn))
        trimap[xn,yn] = 3
      else:
        if trimap[xn,yn] == -1:
          continue
        Q.put((xn,yn))
        trimap[xn,yn] = -1
      
  return 1      
    

for i in range(originalImage.shape[0]):
  for j in range(originalImage.shape[1]):
    if trimap[i,j] == 0:
      ret = traverse(i,j,3)
      if ret==0:
        traverse(i,j,-1)

trimap[mlMask_n==1] = 0
trimap[trimap==3] = 1
trimap[trimap==-1] = 2

trimap[trimap!=1] = 0
temp = np.array(trimap)
disk = morph.disk(1)
temp = morph.binary_dilation(temp, disk)
trimap[temp==0] = 255
trimap[trimap==0] = 128
trimap[trimap==1] = 0

# Get fg/bg distances for each pixel from each surface on convex hull
def convex_hull_distance(cvx_hull, pixels):
    d_hull = np.ones(pixels.shape[0]*cvx_hull.equations.shape[0]).reshape(pixels.shape[0],cvx_hull.equations.shape[0])*1000
    for j, surface_eq in enumerate(cvx_hull.equations):
        for i, px_val in enumerate(pixels):
            nhat= surface_eq[:3]
            d_hull[i,j] = np.dot(nhat, px_val) + surface_eq[3]
    return  np.maximum(np.amax(d_hull, axis=1),0)

def mishima_matte(img, trimap):
    h,w,c = img.shape
    bg = trimap == 0
    fg = trimap == 255
    unknown = True ^ np.logical_or(fg,bg)
    fg_px = img[fg]
    bg_px = img[bg]
    unknown_px = img[unknown]

    # Setup convex hulls for fg & bg
    fg_hull = scipy.spatial.ConvexHull(fg_px)
    fg_vertices_px = fg_px[fg_hull.vertices]
    bg_hull = scipy.spatial.ConvexHull(bg_px)
    bg_vertices_px = bg_px[bg_hull.vertices]

    # Compute shortest distance for each pixel to the fg&bg convex hulls
    d_fg = convex_hull_distance(fg_hull, unknown_px)
    d_bg = convex_hull_distance(bg_hull, unknown_px)

    # Compute uknown region alphas and add to known fg.
    alphaPartial = d_bg/(d_bg+d_fg)
    alpha = unknown.astype(float).copy()
    alpha[alpha !=0] = alphaPartial
    alpha = alpha + fg
    return alpha

alpha = mishima_matte(originalImage.astype(float), trimap.astype(float))

alpha[alpha>0.5] = 1
alpha[alpha!=1] = 0

disk = morph.disk(1)
alpha = morph.binary_closing(alpha, disk)
