import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

annotPath = "path to my csv fime containing the bbox coordinates"
savePath = "path to a folder for images - results"

dfObj = pd.read_csv(annotPath) # read the annotation file

IDs = dfObj['filePath'].unique().tolist() # create a list of paths to image files

for ID in tqdm(IDs):
    try:
        img = cv2.imread(ID) # read image
        fileName = os.path.splitext(os.path.split(ID)[1])[0] # extract file name from path
        # extract annotations from file and draw on image
        for count, (idx, row) in enumerate(dfObj.iterrows()):
            filePath = row["filePath"]
            if filePath == ID:
                label = row["label"]
                x1 = row["startX"]
                y1 = row["startY"]
                x2 = row["endX"]
                y2 = row["endY"]
                top = (int(x1),int(y1))
                bottom = (int(x2),int(y2))
                cv2.rectangle(img, top, bottom, (0, 255, 0), 2)
        # write the image to disk
        cv2.imwrite(savePath+'/{}.jpg'.format(fileName), img)
    except Exception as e:
                print(e)
                pass
