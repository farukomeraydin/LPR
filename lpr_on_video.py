import os
import cv2 as cv
from LPR import LPR
import numpy as np


lpr = LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')

lpr.detect_video('garage2.mkv', kernel=None, minNeighbors=3, isWrite=False, isDisplay=False, minSize=(80, 200), maxSize=(200, 400), printFPS=False)

