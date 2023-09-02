import math

def iou(a,b):
    '''
        Get the intersection over union IoU distance between two bounding boxes a and b
        Each bounding box is a list (x,y,w,h)
        where (x,y) is the top-left corner
        and (w,h) is the width and height
        In general an IoU of >= 0.5 is considered to be a match'''
    
    intersection = rectint(a,b)
    union = (a[2] * a[3]) + (b[2] * b[3]) - intersection
    return intersection / (union + 1e-7)

def rectint(a, b):  
    # return overlapping area of rectangles
    # returns 0 if rectangles don't intersect
    # a and b are rectangles (x,y,w,h)

    a_xmax = a[0] + a[2]
    a_xmin = a[0]
    a_ymax = a[1] + a[3]
    a_ymin = a[1]

    b_xmax = b[0] + b[2]
    b_xmin = b[0]
    b_ymax = b[1] + b[3]
    b_ymin = b[1]

    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0
