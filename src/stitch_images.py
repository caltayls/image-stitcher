import numpy as np
import cv2
from src.utils import crop_image, remove_background
import matplotlib.pyplot as plt
import glob
import re
import shutil
import os

class Stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False, inter_flag=0, descriptor='orb'):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
        imageA, imageB = images
        kpsA, featuresA = self.detectAndDescribe(imageA, descriptor)
        kpsB, featuresB = self.detectAndDescribe(imageB, descriptor)
        # match features between the two images
        matches, H, status = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh, descriptor)
        
        result = cv2.warpPerspective(
            src=imageB,
            M=H,
            dsize=(imageA.shape[1] + imageB.shape[1], imageA.shape[0] + imageB.shape[0]),
            borderMode=5,
            flags=inter_flag
        )
  
        result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA


        # check to see if the keypoint matches should be visualized
        if showMatches:
            self.drawMatches(imageA, imageB, kpsA, kpsB, matches)
    
        return result

    def detectAndDescribe(self, image, descriptor):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # uncomment
        except:
            print('already gray')
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
    

def stitch_show(images, is_path=True, ratio=0.7, reprojThresh=5, inter_flag=4, descriptor='orb', save_as=None, showMatches=False):
    if is_path:
        images = [cv2.imread(image_path) for image_path in images]

    # if not flipped:
    stitched = Stitcher().stitch(images, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag)
        # cropped = crop_image(stitched, path=False)
  
    # else:
    images_flipped = [cv2.flip(image, 1) for image in images[-1::-1]]
    stitched_flipped = Stitcher().stitch(images_flipped, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
    flip_back = cv2.flip(stitched_flipped, 1)
        # cropped = crop_image(flip_back, path=False)
  


    plot2(stitched, flip_back)

    while True:
        print('normal or flipped?')
        user_inp = input()

        if user_inp == 'normal':
            flipped = False
            img = crop_image(stitched, path=False)
            break
        elif user_inp == 'flipped':
            flipped = True
            img = crop_image(flip_back, path=False)
            break

    if save_as is not None:
        img = remove_background(img)
        cv2.imwrite(f'{save_as}.png', img)
    


def plot2(im1, im2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(im1)
    ax2.imshow(im2)
    plt.show()


paths = glob.glob(r'images\battersea\stitched\52\stitched/*')
paths = sorted(paths, key=os.path.getctime)

pattern = re.compile("(\d+_\d+_?)+")

stitch_show(
    [r"images\battersea\stitched\53\stitched.png", r"images\battersea\raw\53\5308.png"],
    save_as=f'images/battersea/stitched/53/stitched'
)


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

stitch_all(r"images\battersea\raw\52", r"images\battersea\stitched\52")


stitch_show(
    [r"images\battersea\stitched\52\stitched.png", r"images\battersea\stitched\53\stitched.png"],
    save_as=f'images/battersea/stitched/52-53'
)

