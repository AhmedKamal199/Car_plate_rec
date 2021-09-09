import time
import cv2
import numpy as np
from PIL import Image
import subprocess as sub
def imageProcess():
	##number plate localization and background delete
	# Importing NumPy,which is the fundamental package for scientific computing with Python
	global start_time
	start_time=time.time()
	# Reading Image
	#img = cv2.imread("image.jpg")

	#img = imga[900:1700, 500:1700] #can be restrict to smaller region
	im = Image.open('image.jpg').convert('L')
	left = 135
	top = 143
	right = 245
	bottom = 175

	im = im.crop((left,top, right,bottom))
	im.save('00.png')
	cmd = "tesseract 00.png out --psm 7"
	print(sub.check_output(cmd, shell=True))
	#cv2.imwrite("1.jpg", img)


imageProcess()