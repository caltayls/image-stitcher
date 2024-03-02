import numpy as np
import cv2
from src.crop_image import crop_image
import matplotlib.pyplot as plt
import glob
import skimage


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
    


def stitch_show(image_paths, is_path=True, flipped=False, ratio=0.7, reprojThresh=5, inter_flag=4, descriptor='orb', save_as=None, showMatches=False):
    if flipped:
        images = [cv2.flip(cv2.imread(image_path), 1) for image_path in image_paths[-1::-1]]

        stitched = Stitcher().stitch(images, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag, )
        flip_back = cv2.flip(stitched, 1)
        cropped = crop_image(flip_back, path=False)
    else:
        if is_path:
            images = [cv2.imread(image_path) for image_path in image_paths]
        else:
            images = image_paths
        stitched = Stitcher().stitch(images, descriptor='orb', ratio=ratio, reprojThresh=reprojThresh, showMatches=showMatches, inter_flag=inter_flag)
        cropped = crop_image(stitched, path=False)

    # cv2.imshow('stitched', cropped)
    # cv2.waitKey()
    # plt.imshow(cropped), plt.show()
    
    if save_as is not None:
        cv2.imwrite(f'{save_as}.jpg', cropped)
    



def stitch_multiple(paths, flipped=False, ratio=0.7, reprojThresh=5):
    for i in range(len(paths)-1):
        image_paths = paths[i:i+2]
        image1_name = image_paths[0].split('\\')[-1].split('.')[0]
        image2_name = image_paths[1].split('\\')[-1].split('.')[0]
        file_name = f"{image1_name}-{image2_name}"
        if flipped:
            file_name = file_name + "_flipped"
        stitch_show(image_paths, descriptor='orb', flipped=flipped, ratio=ratio, reprojThresh=reprojThresh, showMatches=True, save_as=file_name)
       

def plot2(im1, im2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(im1)
    ax2.imshow(im2)
    plt.show()




if __name__ == '__main__':

    paths = glob.glob(r'images\battersea\raw\*')
    # paths = [r"images\battersea\raw\5304.png", r"images\tyneside_1953_aerial\raw\5305.png"]
    # paths = [r"C:\Users\callu\OneDrive\Pictures\jarrow_oblique\factory\facing_ya\im1.png", r"C:\Users\callu\OneDrive\Pictures\jarrow_oblique\factory\facing_ya\im2.png"]

    # stitch_multiple(paths, flipped=True, ratio=0.9, reprojThresh=5)
    

    stitch_show(paths, showMatches=False, save_as=f'0405')

