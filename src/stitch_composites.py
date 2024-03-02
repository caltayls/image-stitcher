import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import try_all_threshold, sobel
from skimage import exposure
import skimage

# Load the images
image1 = cv.imread(r'images\1947_south_tyneside\1182.png')
image2 = cv.imread(r'images\1947_south_tyneside\1183.png')

# Convert images to grayscale
image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# plt.hist(gray1.ravel(), color='blue', alpha=0.6)
# plt.hist(gray2.ravel(), color='orange', alpha=0.6)
# plt.show()
def plot2(im1, im2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(im1)
    ax2.imshow(im2)
    plt.show()



# Initiate SIFT detector

sift = cv.SIFT_create()
 
# find the keypoints and descriptors with SIFT
kps1, features1 = sift.detectAndCompute(image1,None)
kps1[0].queryIdx
kps2, features2 = sift.detectAndCompute(image2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(features1, features2 ,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
refined_matches = []
indices = []


# ratio test as per Lowe's paper
for i, match in enumerate(matches):
 if match[0].distance < 0.7*match[1].distance:
    refined_matches.append((match[0].trainIdx, match[0].queryIdx))
    matchesMask[i]=[1,0]
    indices.append(i)


ptsA = np.float32([kps1[i].pt for (_, i) in refined_matches])
ptsB = np.float32([kps2[i].pt for (i, _) in refined_matches])

# first two are wrong
# refined_matches = refined_matches[2:]
# matchesMask = np.array(matchesMask)
# matchesMask[indices[:2]] = [0,0]
# indices = indices[2:]
H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, 4.0)
image1
result = cv.warpPerspective(image1, H,
        (image1.shape[1] + image2.shape[1], image1.shape[0] + image2.shape[0]))
result[0:image2.shape[0], 0:image2.shape[1]] = image2
result
cv.imshow('qwerty', result)
cv.waitKey()

# for i in indices:
#     matchesMaskEdit = np.array(matchesMask)
#     matchesMaskEdit[:] = [0,0]
#     matchesMaskEdit[i] = [1,0]
draw_params = dict(
matchColor = (0,255,0),
singlePointColor = (255,0,0),
matchesMask = matchesMask,
flags = cv.DrawMatchesFlags_DEFAULT
)

img3 = cv.drawMatchesKnn(image1,kps1,image2,kps2,matches,None, **draw_params)
draw_matches = plt.imshow(img3,),plt.show()

# Step 5: Compute the homography matrix
# src_pts = np.float32([kps1[m.queryIdx].pt for m in refined_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([kps2[m.trainIdx].pt for m in refined_matches]).reshape(-1, 1, 2)

# Step 6: Warp and stitch the images together

# plot2(result, draw_matches)


