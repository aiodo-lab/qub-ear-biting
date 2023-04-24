# Use to pre-process a video e.g., resize, mask frames.

# RUN as shown below: 

# python preprocess_videos.py --input video/5min_04-09-2020_10-42_to_11-56.mp4 --output 5min_04-09-2020_10-42_to_11-56.avi 


# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
args = vars(ap.parse_args())

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# mask upper part of video frame 
	mask = np.zeros(frame.shape[:2], dtype = "uint8")
	cv2.rectangle(mask, (0, 200), (1920, 1080), 255, -1) 
	frame = cv2.bitwise_and(frame, frame, mask = mask)
  
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
	# write the output frame to disk
	writer.write(frame)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
