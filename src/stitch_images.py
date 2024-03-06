import numpy as np
import pandas as pd
import cv2
from src.utils import crop_image, remove_background
import matplotlib.pyplot as plt
import glob
import re
import shutil
import os

class Stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False, inter_flag=0, descriptor='orb'):
        imageA, imageB = images
        kpsA, featuresA = self.detectAndDescribe(imageA, descriptor)
        kpsB, featuresB = self.detectAndDescribe(imageB, descriptor)
        # match features between the two images
        matches, H, status = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh, descriptor)
        
        imageB_warped = cv2.warpPerspective(
            src=imageB,
            M=H,
            dsize=(imageA.shape[1] + imageB.shape[1] + 2000, imageA.shape[0] + imageB.shape[0] + 2000),
            borderMode=5,
            flags=inter_flag
        )
  
        combined_image = imageB_warped.copy()
        combined_image[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

        # Remove unwanted black pixels caused by superimposing A onto B
        combined_image_imprvd = self.patch_black_cells(imageA, imageB_warped, combined_image)


        # check to see if the keypoint matches should be visualized
        if showMatches:
            self.drawMatches(imageA, imageB, kpsA, kpsB, matches)
    
        return combined_image_imprvd

    def detectAndDescribe(self, image, descriptor):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # uncomment
        except:
            gray = image

        # detect and extract features from the image
        if descriptor == 'sift':
            detect = cv2.SIFT_create()
        elif descriptor == 'orb':
            detect = cv2.ORB_create(10000)
            kps = detect.detect(image, None)
            descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
            kpts, desc = descriptor.compute(image, kps)
            return kpts, desc

        # (kps, features) = detect.detectAndCompute(gray, None) # change image
        # return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh, descriptor):

        if descriptor == 'sift':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        if descriptor == 'orb':
            FLANN_INDEX_LSH = 6
            index_params= dict(
                algorithm = FLANN_INDEX_LSH,
                table_number = 12,#6, 
                key_size = 20,#12, 
                multi_probe_level = 2 #1
            )

        search_params = dict(checks=100) # or pass empty dictionary

        matcher = cv2.FlannBasedMatcher(index_params,search_params)
        # matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = [match[0] for match in rawMatches if (len(match) == 2) and (match[0].distance < match[1].distance * ratio)]

        # computing a homography requires at least 4 matches
        matches_len = len(matches)
        print(f"{matches_len} matches.")
    
        matches = sorted(matches,key=lambda x:x.distance)
        matches = matches[:100]
        ptsA = np.float32([kpsB[match.trainIdx].pt for match in matches])
        ptsB = np.float32([kpsA[match.queryIdx].pt for match in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return matches, H, status

    def drawMatches(self, image1, image2, kps1, kps2, matches):
        output_image = cv2.drawMatches(image1,kps1,image2, kps2,matches[:100],None,flags=2)
        cv2.imshow('Output image',output_image)
        # return vis

        
    def patch_black_cells(self, imageA, warped, combined):

        # black cells in imgA to replace
        a_black_cells = np.argwhere(imageA == 0)
        img1_black = pd.DataFrame(a_black_cells)

        # indices of warped image B
        b_indices = np.argwhere(warped != 0)
        b_cells = pd.DataFrame(b_indices)


        shared = pd.merge(img1_black, b_cells, how='inner').values

        for i, j in shared:
            combined[i, j] = warped[i, j]

                    
        return combined
    

def stitch_show(images, is_path=True, ratio=0.7, reprojThresh=5, inter_flag=4, descriptor='orb', save_as=None, showMatches=False):
    if is_path:
        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  for image_path in images]

    stitcher = Stitcher()
    # if not flipped:
    im1 = stitcher.stitch(images, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag)
    im1 = crop_image(im1)

    # else:
    images_flipped = [cv2.flip(image, 1) for image in images[-1::-1]]
    stitched_flipped = stitcher.stitch(images_flipped, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
    im2 = cv2.flip(stitched_flipped, 1)
    im2 = crop_image(im2)

    


    # image 2 first
        # if not flipped:
    im3 = stitcher.stitch(images[-1::-1], descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag)
    im3 = crop_image(im3)

  
    # else:
    images_flipped = [cv2.flip(image, 1) for image in images]
    stitched_flipped = stitcher.stitch(images_flipped, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
    im4 = crop_image(cv2.flip(stitched_flipped, 1))



    plot2(im1, im2, im3, im4)

    while True:
        print('Image 1, 2, 3, or 4?')
        user_inp = input()

        if user_inp == '1':
            img = crop_image(im1, path=False)
            break
        elif user_inp == '2':
            img = crop_image(im2, path=False)
            break
        elif user_inp == '3':
            img = crop_image(im3, path=False)
            break
        elif user_inp == '4':
            img = crop_image(im4, path=False)
            break

    if save_as is not None:
        # img = remove_background(img)
        cv2.imwrite(f'{save_as}.png', img)
    


def plot2(im1, im2, im3, im4):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(im3)
    ax4.imshow(im4)
    plt.show()


def stitch_all(image_folder, destination):
    paths = glob.glob(f'{image_folder}/*')

    path_pattern = re.compile("\d{4}")
    order = [int(path_pattern.search(path).group(0)) for path in paths]
    indices_sorted = np.argsort(order)
    paths = [paths[i] for i in indices_sorted]


    shutil.copy(paths[0], f"{destination}/stitched.png")
    
    for path in paths[1:]:
        print(path)
        stitch_show(
            [f"{destination}/stitched.png", path],
            save_as=f"{destination}/stitched"
        )




# stitch_show(
#     [r"images\battersea\raw\53\5307.png", r"images\battersea\raw\52\5211.png"],
#     save_as=f'images/battersea/stitched/5307-5211'
# )


# stitch_show(
#     [r"images\battersea\raw\52\5209.png", r"images\battersea\stitched\5305-08-5209-12.png", ],
#     save_as=r'images\battersea\stitched\5305-08-5209-13'
# )


stitch_show([r"images\battersea\raw\53\5305.png", r"images\battersea\raw\53\5304.png"], save_as=r'images\battersea\stitched\test111')