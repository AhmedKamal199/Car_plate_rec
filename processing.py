import time
import cv2
import numpy as np
from PIL import Image

def imageProcess():
  ##number plate localization and background delete
  # Importing NumPy,which is the fundamental package for scientific computing with Python
  global start_time
  start_time=time.time()
  # Reading Image

  #img_0 = cv2.imread("image.jpg") # old code

  # new code
  im = Image.open('sample3.jpg').convert('L')

  left = 900
  top = 500
  right = 1700
  bottom = 1700

  im = im.crop((left,top, right,bottom))
  im.save('00.png')
  img = cv2.imread('00.png')
  cv2.imwrite('display0.jpg',img)

  # RGB to Gray scale conversion
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Noise removal with iterative bilateral filter(removes noise while preserving edges)
  noise_removal = cv2.bilateralFilter(img_gray,9,75,75)

  # Histogram equalisation for better results
  equal_histogram = cv2.equalizeHist(noise_removal)


  # Morphological opening with a rectangular structure element
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)

  # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
  sub_morp_image = cv2.subtract(equal_histogram,morph_image)

  # Thresholding the image
  ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)


  # Applying Canny Edge detection
  canny_image = cv2.Canny(thresh_image,250,255)

  canny_image = cv2.convertScaleAbs(canny_image)

  # dilation to strengthen the edges
  kernel = np.ones((3,3), np.uint8)
  # Creating the kernel for dilation
  dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

  # Finding Contours in the image based on edges
  contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
  # Sort the contours based on area ,so that the number plate will be in top 10 contours
  screenCnt = None
  # loop over our contours
  loop = 1

    #addition
  #########################
  top_idx:int
  bottom_idx:int
  right_idx:int
  left_idx:int

  top = 0
  bottom = 0
  right = 0
  left = 0

  #########################

  for c in contours:
    print ("loop #: " + str(loop))
    loop = loop+1
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
    print ("approx: " + str(len(approx)))



    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:  # Select the contour with 4 corners
      screenCnt = approx
      top_left = approx[0][0] #[x y]
      top_right = approx[1][0]
      bottom_left = approx[2][0]
      bottom_right = approx[3][0]

      #########################

      top_idx = min(top_left[1], top_right[1])
      bottom_idx = max(bottom_left[1], bottom_right[1])
      left_idx=min(min(top_left[0], top_right[0]),min(bottom_left[0], bottom_right[0]))
      right_idx=max(max(top_left[0], top_right[0]),max(bottom_left[0], bottom_right[0]))
      print ("Yay, find one")
      top = top_idx
      right = right_idx
      left = left_idx
      bottom = bottom_idx
      print(top)
      break

  ## Masking the part other than the number plate
  print(top)
  mask = np.zeros(img_gray.shape,np.uint8)
  new_image = cv2.drawContours(mask,screenCnt,0,255,-1,)
  new_image = cv2.bitwise_and(img,img,mask=mask)

  # new line
  cv2.imwrite('ForFinal.png', new_image)
  # Histogram equal for enhancing the number plate for further processing
  y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_BGR2YCR_CB))
  # Converting the image to YCrCb model and splitting the 3 channels
  y = cv2.equalizeHist(y)
  # Applying histogram equalisation
  final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCR_CB2BGR)
  # Merging the 3 channels

  #cv2.namedWindow("12_Extract",cv2.WINDOW_NORMAL)
  #print(new_image.shape)

  # old code
  ########################################################################
  #final_new_image = new_image[top_idx:bottom_idx,left_idx:right_idx ]
  #########################################################################

  # new code
  #########################################################
  im = Image.open('ForFinal.png').convert('L')


  im = im.crop((left,top, right,bottom))
  im.save('final_image.jpg')
  final_new_image = cv2.imread('final_image.jpg')
  ##########################################################
  #print(final_new_image.shape)
  #cv2.imshow("12_Extract", final_new_image)

  cv2.imwrite('result1.jpg',new_image)
  cv2.imwrite('result2.jpg',final_new_image)

  im = final_new_image
  im[np.where((im <[20,20,20]).all(axis = 2))] = [255,255,255]

  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
  binl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
  open_out = cv2.morphologyEx(binl, cv2.MORPH_OPEN, kernel)
  cv2.bitwise_not(open_out, open_out)
  #cv2.namedWindow("Transfered",cv2.WINDOW_NORMAL)
  #cv2.imshow("Transfered", open_out)
  cv2.imwrite('output1.jpg', open_out)

imageProcess()