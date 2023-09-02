# This code solves a LAP (Linear assignment problem) to associate ear biting events with pig in each frame
# it can handle multiple events and multiple pigs in each frame
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
# https://en.wikipedia.org/wiki/Assignment_problem

from scipy.optimize import linear_sum_assignment
import numpy as np
import csv
import os
from rectint import *
import pandas as pd

# Data paths
imgPath = "C:/Users/3057107/pig_tracker_axis_aligned/QUB/test/Seq-rm4pen6/img1"
pig_annotPath = "D:/mydocs/QUB-updated/GII/Shared with Niall/tracker_output_CSV_8_august_pig/tracker_output_pigs/Seq-rm4pen6.csv"
bite_annotPath = "D:/mydocs/QUB-updated/GII/Shared with Niall/tracker_output_pigs_ear_biting_events/tracker_output_pigs_ear_biting_events/Seq-rm4pen6.csv"

''' Load the csv files
    and determine the number of frames in the sequence
'''
all_pig_detections = np.loadtxt(pig_annotPath, delimiter=',')
all_eb_event_detections = np.loadtxt(bite_annotPath, delimiter=',')
imgList = os.listdir(imgPath)
# print(imgList)
frames_in_video = len([int(i.split('.', 1)[0]) for i in imgList])
                                
MAX_ASSIGNMENT_COST = 100

df = pd.DataFrame()
for frame in range(frames_in_video): 
    # get all the pig bboxes and ear-biting bboxes that happen in this frame
    # I assume pig_detections_this_frame and ear_biting_events_this_frame are numpy arrays
    # with [frame,track_number,x,y,w,h]
    bite_bbox_locs = np.where(all_eb_event_detections[:,0] == frame)
    ear_biting_events_this_frame = all_eb_event_detections.take(bite_bbox_locs, axis=0)[0]
    ear_biting_events_this_frame = ear_biting_events_this_frame.tolist()
    num_events = len(ear_biting_events_this_frame)
    #num_events = np.shape(ear_biting_events_this_frame)[1]
    pig_bbox_locs = np.where(all_pig_detections[:,0] == frame)
    pig_detections_this_frame = all_pig_detections.take(pig_bbox_locs, axis=0)[0]
    pig_detections_this_frame = pig_detections_this_frame.tolist()
    num_pigs = len(pig_detections_this_frame)
    #num_pigs = np.shape(pig_detections_this_frame)[1]
    # a matrix to hold the IOU between pig bboxes and EB events	
    # this will be used to find the assignments between pigs and EB events 	
    # each entry holds and assignment cost between pigs and EB events 	
    # initially we give them all a high cost 	
    # Note - very important to ensure the matrix dimensions are the correct way around 	
    # i.e. events, then pigs in order for the assignment to work. I've tried to keep this 	
    # consistent across the rest of the code
    association_matrix = np.ones((num_events,num_pigs)) * MAX_ASSIGNMENT_COST
    
    pig_detections_this_frame_x = np.array(pig_detections_this_frame) 
    ear_biting_events_this_frame_x = np.array(ear_biting_events_this_frame)
    for i in range(num_events): 
        event_bbox = ear_biting_events_this_frame_x[i,2:6] 
        for j in range(num_pigs): 
            pig_bbox = pig_detections_this_frame_x[j,2:6] 
            # the assignment cost is (1 - IOU) between the pig and ear-biting event bbox
            # we may need to use an if-statement here to only allow cases there IOU > 0.5 
            # to be sure that the pig is associated with the event... We can experiment with this...
            # Also ... 
            # IOU may not work well since it assumes both bboxes are similar in size. But in fact 
            # the EB bbox is much smaller. We might need to use the intersection with the EB box instead. 
            # Again more experiments needed...
            # iouNow = iou(event_bbox, pig_bbox)
            # print(iouNow)
            association_matrix[i,j] = 1-iou(event_bbox, pig_bbox) 
    # now we solve a linear-assignment problem to find the optimal assignment
    # between pig bboxes and event bboxes. We want to find the set of assignments
    # with minimum cost  
    # print(association_matrix)     
    row_ind, col_ind = linear_sum_assignment(association_matrix)
    # print(association_matrix[row_ind, col_ind].sum())
    for r,c in list(zip(row_ind, col_ind)):
        if not association_matrix[r][c] == MAX_ASSIGNMENT_COST:
            record_association = (pig_detections_this_frame_x[c], ear_biting_events_this_frame_x[r])
            # with open('record_association_rm5pen2.csv','w', newline='') as my_csv:
            #         csvWriter = csv.writer(my_csv,delimiter=',')
            #         csvWriter.writerows(record_association)
            # print(record_association)
            frame = record_association[1][0]
            event_ID = record_association[1][1]
            event_minx = record_association[1][2]
            event_miny = record_association[1][3]
            event_maxx = record_association[1][2] + record_association[1][4]
            event_maxy = record_association[1][3] + record_association[1][5]
            pig_ID = record_association[0][1]
            pig_minx = record_association[0][2]
            pig_miny = record_association[0][3]
            pig_maxx = record_association[0][2] + record_association[0][4]
            pig_maxy = record_association[0][3] + record_association[0][5]
            
            dictn = {'frame': frame, 'event_ID': event_ID, 'event_minx':event_minx, 'event_miny':event_miny, 'even_maxx':event_maxx, 'event_maxy':event_maxy,
                     'pig_minx':pig_minx, 'pig_miny':pig_miny, 'pig_maxx':pig_maxx, 'pig_maxy':pig_maxy, 'pig_ID':pig_ID}
            df = df.append(dictn, ignore_index=True)
    #     # write the image to disk
# cv2.imwrite(savePath+'/{}.jpg'.format(fileName), image)
df.to_csv(r"D:\mydocs\QUB-updated\GII\Shared with Niall\assoc-axis-align-ear-pig\record_association_rm4pen6.csv", header=True)            
