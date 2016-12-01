#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
# from video import create_capture
# from common import clock, draw_str

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def crop_img(img, rects):
    for x1, y1, x2, y2 in rects:
	return img[y1:y2, x1:x2]

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/nariz.xml")

    # added
    if (len(sys.argv) < 2):
	sys.exit()

    imgName = sys.argv[1]

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    # added
    i = 0

    while (i < 100):
	# img = cv2.imread('10.jpg', 1) #load image
	JPG = str(i) + '.PNG'
	PGM = str(i) + '.pgm'	

	img = cv2.imread(JPG, 1)	

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	rects = detect(gray, cascade)
 	vis = crop_img(gray, rects) #remove background	
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #contrast
	vis = clahe.apply(vis)
	vis = cv2.resize(vis, (100, 100)) #resize

	cv2.imshow('facedetect', vis)
	cv2.imwrite(PGM, vis)

	i += 1

	if 0xFF & cv2.waitKey(5) == 27:
            break
	
    cv2.destroyAllWindows()
