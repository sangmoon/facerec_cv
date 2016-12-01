#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, PCA, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import MyValidation as MV
from facerec.util import minmax_normalize
from facerec.serialization import load_model
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import logging

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
        print "USAGE: facerec_demo.py </path/to/test_images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X, y] = read_images(sys.argv[1])
    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    model = load_model('myModel.pkl')

    cv = MV(model)
    cv.validate(X, y)
    cv.print_results()
