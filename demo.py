#!/usr/bin/python

import torch
import argparse
import sys
import cv2
import numpy as np
import time
import demo_utils.vot as vot
from demo_utils.siamvggtracker import SiamVGGTracker


# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************

parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--model', metavar='model', default='SiamFCNext22', type=str,
                    help='which model to use.')

args = parser.parse_args()

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
tracker = SiamVGGTracker(args.model, imagefile, selection)

if not imagefile:
    sys.exit(0)

toc = 0

while True:
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************

    tic = cv2.getTickCount()

    imagefile = handle.frame()
    image = cv2.imread(imagefile)
    if not imagefile:
        break
    region, confidence = tracker.track(imagefile)
    toc += cv2.getTickCount() - tic

    region = vot.Rectangle(region.x, region.y, region.width, region.height)
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)
    cv2.rectangle(image, (int(region.x), int(region.y)), (int(region.x + region.width), int(region.y + region.height)), (0, 255, 255), 3)
    cv2.imshow('SiameseX', image)
    cv2.waitKey(1)
    # if cv2.waitKey() == 27:
    #     break

    print('Tracking Speed {:.1f}fps'.format((len(handle) - 1) / (toc / cv2.getTickFrequency())))

