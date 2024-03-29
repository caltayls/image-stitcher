import numpy as np
import pandas as pd
import cv2
try:
    from src.utils import crop_image, remove_background
except:
    from utils import crop_image, remove_background

import matplotlib.pyplot as plt
import glob
import re
import shutil
import os

class Stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False, inter_flag=4, descriptor='orb', new_image_on_top=True):
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
        combined_image = self.patch_black_cells(imageA, imageB_warped, combined_image, b_on_top=new_image_on_top)


        # check to see if the keypoint matches should be visualized
        if showMatches:
            self.drawMatches(imageA, imageB, kpsA, kpsB, matches)
    
        return combined_image

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
        output_image = cv2.drawMatches(image1,kps1,image2, kps2,matches[:100], None, flags=2)
        cv2.imshow('Output image', output_image)
        # return vis

        
    def patch_black_cells(self, imageA, warped, combined, b_on_top=True):


        # indices of warped image B
        b_indices = np.argwhere(warped != 0)
        
        if b_on_top:
            combined[b_indices[:, 0], b_indices[:, 1]] = warped[b_indices[:,0], b_indices[:,1]]
            return combined
    
        # black cells in imgA to replace
        a_black_cells = np.argwhere(imageA == 0)
        img1_black = pd.DataFrame(a_black_cells)
        b_cells = pd.DataFrame(b_indices).values
        shared = pd.merge(img1_black, b_cells, how='inner').values
        combined[shared[:, 0], shared[:, 1]] = warped[shared[:,0], shared[:,1]]

        return combined
    

def stitch_show(images, is_path=True, automate=True, ratio=0.7, reprojThresh=5, inter_flag=4, descriptor='orb', save_as=None, showMatches=False, inside=False, new_image_on_top=True):
    if is_path:
        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  for image_path in images]
    
    
  
    image_sizes = [image.shape[0] * image.shape[1] for image in images]
    size_indices = np.argsort(image_sizes, )
    images = [images[i] for i in size_indices[-1::-1]]
    processed_images = []
    stitcher = Stitcher()
    #determine order of merge:
    im1 = stitcher.stitch(images, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, new_image_on_top=new_image_on_top)
    im1 = crop_image(im1)
    if inside:
        cv2.imwrite(save_as, im1)
        return

    
    processed_images.append(im1)


    images_flipped = [cv2.flip(image, -1) for image in images]
    stitched_flipped = stitcher.stitch(images_flipped, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
    im2 = cv2.flip(stitched_flipped, -1)
    im2 = crop_image(im2)
    processed_images.append(im2)



    v_flip = [cv2.flip(image, 0) for image in images]
    im3 = stitcher.stitch(v_flip, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag)
    im3 = cv2.flip(im3, 0)
    im3 = crop_image(im3)
    processed_images.append(im3)


    images_flipped = [cv2.flip(image, 1) for image in images]
    stitched_flipped = stitcher.stitch(images_flipped, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
    im4 = cv2.flip(stitched_flipped, 1)
    im4 = crop_image(im4)
    processed_images.append(im4)


    if automate:
        image_sizes = [image.shape[0] * image.shape[1] for image in processed_images]
        largest_image = np.argmax(image_sizes)
        cv2.imwrite(save_as, processed_images[largest_image])


    else:
        show_images(im1, im2, im3, im4)

        while True:
            print('Image 1, 2, 3, or 4?')
            user_inp = input()

            if user_inp == '1':
                img = im1
                break
            elif user_inp == '2':
                img = im2
                break
            elif user_inp == '3':
                img = im3
                break
            elif user_inp == '4':
                img = im4
                break

        if save_as is not None:
            # img = remove_background(img)
            cv2.imwrite(save_as, img)
    


def show_images(im1, im2, im3, im4):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(im1)
    ax1.set_title('Image 1')
    ax2.imshow(im2)
    ax2.set_title('Image 2')
    ax3.imshow(im3)
    ax3.set_title('Image 3')
    ax4.imshow(im4)
    ax4.set_title('Image 4')
    plt.show()


def stitch_all(paths, destination, is_list=False, automate=True):
    if not is_list:
        paths = glob.glob(paths)
        path_pattern = re.compile(r"\d{4}")
        order = [int(path_pattern.findall(path)[-1]) for path in paths]
        print(order)
        indices_sorted = np.argsort(order)
        paths = [paths[i] for i in indices_sorted]

    shutil.copy(paths[0], destination)
    
    for i, path in enumerate(paths[1:]):
        print(f"Combining:\n-- {paths[i]}\n-- {paths[i+1]}")
        stitch_show([destination, path], save_as=destination, automate=automate)


# for i in range(5183, 5207):

#     n = (i - 5183) // 4

#     if (i - 5183) % 4 == 0:
#         print(n)
#         stitch_show(
#             [ fr'images\battersea\raw\2103_v_{i}.png',fr'images\battersea\raw\2103_v_{i+1}.png',],
#             save_as=rf'images\battersea\stitched\belgravia_{n}.png',
#             automate=True
#         )
#     else:
#         stitch_show(
#             [ fr'images\battersea\stitched\belgravia_{n}.png' ,fr'images\battersea\raw\2103_v_{i+1}.png',],
#             save_as=rf'images\battersea\stitched\belgravia_{n}.png',
#             automate=True
#         )


# for i in range(6):
#     if i == 0:

#         stitch_show(
#         [ fr'images\battersea\stitched\belgravia_{i}.png' ,fr'images\battersea\stitched\belgravia_{i+1}.png'],
#         save_as=rf'images\battersea\stitched\belgravia.png',
#         automate=True
#         )

#     else:
#         stitch_show(
#         [ fr'images\battersea\stitched\belgravia.png' ,fr'images\battersea\stitched\belgravia_{i}.png',],
#         save_as=rf'images\battersea\stitched\belgravia.png',
#         automate=True
#         )


stitch_show(
[ fr'images\battersea\stitched\thames_master_41.png' ,fr'images\battersea\stitched\5219-5195_6.png'],
save_as=rf'images\battersea\stitched\thames_master_42.png',
automate=False,
inside=True
)

# import time

# t0 = time.time()

# for i in range(10):
#     stitch_show(
#     [ fr'images\battersea\stitched\thames_master_{41+i}.png' ,fr'images\battersea\raw\2103_v_52{17+i}.png'], #5120
#     save_as=rf'images\battersea\stitched\thames_master_{42+i}.png',
#     automate=False,
#     inside=True
#     )
# dt = time.time() - t0
# print(dt)

# paths = glob.glob(r"images\battersea\raw\*v_50*")
# path_pattern = re.compile(r"\d{4}")
# order = np.array([int(path_pattern.findall(path)[-1]) for path in paths])
# paths_new = np.array(paths)[(order > 5034) & (order < 5053) ]
# paths_new

# stitch_all(paths_new, r"images\battersea\stitched\westminster_east1.png", is_list=True, automate=False)
# for i, path in enumerate(paths_new):
#     stitch_show(
#     [f'images/battersea/stitched/thames_master_{12+i}.png', path,],
#     save_as=f'images/battersea/stitched/thames_master_{13+i}.png',
#     automate=False
# )

