#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, PCA, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor, SVM
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation as KFCV, MyValidation as MV
from facerec.visual import subplot
from facerec.util import minmax_normalize, asRowMatrix
from facerec.serialization import save_model, load_model
from sklearn import preprocessing

# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

def read_images2(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for male_or_female in dirnames:
            male_or_female_path = os.path.join(dirname, male_or_female)
	    for dirname2, dirnames2, filenames2 in os.walk(male_or_female_path):
	        for subdirname in dirnames2:
		    subject_path = os.path.join(dirname2, subdirname)
            	    for filename in os.listdir(subject_path):
                	try:
                    	    im = Image.open(os.path.join(subject_path, filename))
                    	    im = im.convert("L")
                            # resize to given size (if given)
                            if (sz is not None):
                                im = im.resize(sz, Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(c)
                        except IOError, (errno, strerror):
                            print "I/O error({0}): {1}".format(errno, strerror)
                        except:
                            print "Unexpected error:", sys.exc_info()[0]
                            raise
            c = c+1
    return [X,y]

if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images2(sys.argv[1])
    [TX,Ty] = read_images2(sys.argv[2])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = PCA()
    classifier = SVM()
    # Define the model as the combination
    my_model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    TX = scaler.transform(TX)

    my_model.compute(X, y)
    # We then save the model, which uses Pythons pickle module:
    # save_model('myModel.pkl', my_model)

    cv = MV(my_model)
    cv.validate(TX, Ty)
    cv.print_results()
