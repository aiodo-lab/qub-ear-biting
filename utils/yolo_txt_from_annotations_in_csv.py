from tqdm import tqdm
import pandas
import numpy as np

annotPath = "/media/anicetusodo/Data/datasets/npg_data/insulator_detection/defective.csv"
savePath = "/media/anicetusodo/Data/datasets/npg_data/insulator_detection/labels_defects"

dfObj = pandas.read_csv(annotPath)
IDs = dfObj['Source'].unique().tolist()

for ID in tqdm(IDs):
    try:
        fileName = str(ID).zfill(3)  
        bboxes = []
        for count, (idx, row) in enumerate(dfObj.iterrows()):
            filePath = row["Source"]
            if filePath == ID:
                label = row["class"]
                label = str(label)
                x1 = row["Column2"]
                y1 = row["Column3"]
                w = row["Column4"]
                h = row["Column5"]            
                box = (label, x1, y1, w, h) 
                bboxes.append(box)
        # from numpy
        np.savetxt(savePath+"/{}.txt".format(fileName), bboxes, delimiter =" ", fmt ='% s')
    except Exception as e:
        print(e)
        pass
    
