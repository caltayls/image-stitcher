import cv2

def crop_image(image, path=True):
    if path:
        stitched_image = cv2.imread(image)
    else:
        stitched_image = image
    # Convert the stitched image to grayscale
    gray_stitched = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(gray_stitched, 1, 255, cv2.THRESH_BINARY)
    # Find non-zero (foreground) pixels in the binary mask
    non_zero_pixels = cv2.findNonZero(binary_mask)
    # Compute the bounding box that encloses the non-zero pixels
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    # Crop the stitched image based on the bounding box
    cropped_image = stitched_image[y:y+h, x:x+w]

    return cropped_image

def remove_background(image, save=False, save_path=None):
    # Convert image to image gray 
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
    # Applying thresholding technique 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 
      
    # Using cv2.split() to split channels  
    # of coloured image 
    b, g, r = cv2.split(image) 
      
    # Making list of Red, Green, Blue 
    # Channels and alpha 
    rgba = [b, g, r, alpha] 
      
    # Using cv2.merge() to merge rgba 
    # into a coloured/multi-channeled image 
    dst = cv2.merge(rgba, 4) 
      
    if save:
      cv2.imwrite(f"{save_path}.png", dst)

    return dst 