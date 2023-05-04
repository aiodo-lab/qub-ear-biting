import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

annotPath = "/home/anicetusodo/detection-projects/utils/train/pigs(1685).jpg"
img = cv2.imread(annotPath) # read image
x1 = 0
y1 = 198
x2 = 78
y2 = 319
top = (int(x1),int(y1))
bottom = (int(x2),int(y2))
cv2.rectangle(img, top, bottom, (0, 255, 0), 2)
cv2.imshow('ImageWindow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
