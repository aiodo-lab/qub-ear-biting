import os
import cv2
from tqdm import tqdm
import pandas as pd

annotPath = "path to annotation file containing image paths and X1Y1X2Y2 coordinates"
savePath = "path to save labels .txt"

dfObj = pd.read_csv(annotPath)
print(dfObj[:2])

def convert(h, w, x1, x2, y1, y2):
    nw = 1./w
    nh = 1./h
    xi = (x1 + x2)/2.0
    yi = (y1 + y2)/2.0
    wi = x2 - x1
    hi = y2 - y1
    x = xi*nw
    w = wi*nw
    y = yi*nh
    h = hi*nh
    
    return(x, y, w, h)    

IDs = dfObj['filePath'].unique().tolist()

for ID in tqdm(IDs):
    
    df = pd.DataFrame()
    for count, (idx, row) in enumerate(dfObj.iterrows()):
        filePath = row["filePath"]
        if filePath == ID:
            try:
                img = cv2.imread(filePath)
                height = int(img.shape[0])
                width = int(img.shape[1])
                fileName = os.path.splitext(os.path.split(filePath)[1])[0]
                #image = cv2.imwrite(savePath+'/{}.jpg'.format(fileName), img)
                label = row["label"]
                x1 = row["startX"]
                y1 = row["startY"]
                x2 = row["endX"]
                y2 = row["endY"]
                (x, y, w, h) = convert(height, width, x1, x2, y1, y2)
                #print(label, x, y, w, h)
                dictn = {"label": label, "x": x, "y": y, "w": w, "h": h}
                df = df.append([dictn], ignore_index=True)
            except Exception as e:
                print(e)
                pass
            
    df.to_csv(savePath+"/{}.txt".format(fileName), header=False, index=False, sep=' ', mode='w')
