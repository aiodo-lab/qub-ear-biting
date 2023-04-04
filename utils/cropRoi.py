import os
import cv2
import imutils
from tqdm import tqdm
import pandas as pd
import numpy as np

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")
        
cropPath = "trainCrops_040423"
imgPath = "/home/anicetusodo/detection-projects/val-tracker/train-siamese.csv"

df = pd.read_csv(imgPath)

imageList = df['filePath'].unique().tolist()

for fPath in tqdm(imageList):
  try:
    for count, (idx, row) in enumerate(df.iterrows()):
      filePath = row["filePath"]
      if filePath == fPath:
        img = cv2.imread(fPath)
        regID = str(row["region_id"])
        save_path = os.path.join(cropPath,regID)
        create_dir(save_path)
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        x2 = x+w
        y2 = y+h
        crop = img[y:y2, x:x2]
        image = cv2.imwrite(save_path+'/{}.png'.format(idx), crop)
  except OSError:
    print(f"Error: creating directory with name {path}")
