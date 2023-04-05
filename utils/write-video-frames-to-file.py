import os
import numpy as np
import cv2
import imutils
from glob import glob

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def save_frame(video_path, save_dir):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            break
        # mask upper region of each frame to cover adjoining pen
        mask = np.zeros(frame.shape[:2], dtype = "uint8")
        cv2.rectangle(mask, (0, 200), (1920, 1080), 255, -1) 
        newImage = cv2.bitwise_and(frame, frame, mask = mask)
        
        # save the frame to disk 
        cv2.imwrite(f"{save_path}/{idx}.png", newImage)
        idx += 1

if __name__ == "__main__":
    video_paths = glob("videos/*")
    save_dir = "frames_1"
    for path in video_paths:
        save_frame(path, save_dir)
