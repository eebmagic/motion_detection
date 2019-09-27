'''
 LOGIC OUTLINE:

    make average of each pixel from frame
    abs-subtract of current from average pixel-by-pixel frame
    sum of difference from average frame values

    test for threshold

'''
import cv2
import os
import sys
import numpy as np

########################################################################
# SETTINGS #
SHOW_REAL_VIDEO = ("-s" in sys.argv[:] or "--show" in sys.argv[:])   # Set this to True to get real camera video from cv2
SHOW_START_FRAME = ("-i" in sys.argv[:] or "--initial" in sys.argv[:])  # Check for option for first frame

# Max pixel difference over which to trigger the motion alarm
THRESHOLD = 100

########################################################################

def format_frame(inputFrame):
    # Convert to grayscale
    gray = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

    # Return cropped frame
    # Sidewalk
    # return gray[480:-150, 300:-350]
    
    # Street (No Parked Car)
    return gray[0:-650, 455:-560]
    
    # Streen and sidewalk (NO TREES)
    # return gray[0:-150, 450:-560]


def count_different_pixels(frame_a, frame_b):
    pass


# Start of main

cap = cv2.VideoCapture(0)
import time
time.sleep(1)
start_frame = format_frame(cap.read()[1])
print("First Frame:")
print(start_frame)

if SHOW_START_FRAME:
    cv2.imshow("Start Frame", start_frame)

# next_frame = format_frame(cap.read()[1])
# print(next_frame)

# diff = cv2.absdiff(start_frame, next_frame)
# print("\nAbsDifference Frame:")
# print(diff)
# quit()


SUM_FRAME = start_frame.astype(np.uint32)
AVG_FRAME = start_frame.astype(np.uint32)

SUM_CHANGE = 0
AVG_CHANGE = 0

counter = 1

while(cv2.waitKey(1) & 0xFF != ord('q')):
    # Get image data
    frame = format_frame(cap.read()[1])

    # DIFF = np.absolute(AVG_FRAME - frame)
    DIFF = cv2.absdiff(np.float32(AVG_FRAME), np.float32(frame))
    CHANGE = int(DIFF.sum())
    MAX_DIFF = np.amax(DIFF)

    print(counter)
    # print(int(SUM_FRAME.sum()))
    # print(SUM_FRAME)
    # print(int(AVG_FRAME.sum()))
    # print(AVG_FRAME)
    # print(f"frame.sum:  {frame.sum()}")
    print(f"CHANGE:     {CHANGE}")
    print(f"SUM_CHANGE: {SUM_CHANGE}")
    print(f"AVG_CHANGE: {AVG_CHANGE}\n")
    print(f"MAX_DIFF: {MAX_DIFF}\n")

    # Check for change dist from avg
    if MAX_DIFF > THRESHOLD:
        print("\t\t\tMOVEMENT DETECTED!")

    # Check for flukes
    # Update AVG and SUM frames
    # if frame.sum() < 30_000_000:
    counter += 1
    SUM_FRAME = np.add(SUM_FRAME, frame)
    AVG_FRAME = (SUM_FRAME / counter).astype(np.uint8)

    SUM_CHANGE += CHANGE
    AVG_CHANGE = int(SUM_CHANGE / counter)
    
    # Display the resulting frame
    if SHOW_REAL_VIDEO:
        cv2.imshow("Frame", frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()