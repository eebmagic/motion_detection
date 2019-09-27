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
# Set this to True to get real camera video from cv2
SHOW_REAL_VIDEO = ("-s" in sys.argv[:] or "--show" in sys.argv[:])

# Check for option for first frame
SHOW_START_FRAME = ("-i" in sys.argv[:] or "--initial" in sys.argv[:])


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


# Start of main

cap = cv2.VideoCapture(0)
import time
time.sleep(1)
start_frame = format_frame(cap.read()[1])
print("First Frame:")
print(start_frame)

if SHOW_START_FRAME:
    cv2.imshow("Start Frame", start_frame)


SUM_FRAME = start_frame.astype(np.uint32)
AVG_FRAME = start_frame.astype(np.uint32)

counter = 1

while(cv2.waitKey(1) & 0xFF != ord('q')):
    # Get image data
    frame = format_frame(cap.read()[1])

    DIFF = cv2.absdiff(np.float32(AVG_FRAME), np.float32(frame))
    MAX_DIFF = np.amax(DIFF)

    print(counter)
    print(f"MAX_DIFF: {MAX_DIFF}\n")

    # Check for change dist from avg
    if MAX_DIFF > THRESHOLD:
        print("\t\t\tMOVEMENT DETECTED!")

    # Update AVG and SUM frames
    counter += 1
    SUM_FRAME = np.add(SUM_FRAME, frame)
    AVG_FRAME = (SUM_FRAME / counter).astype(np.uint8)
    
    # Display the resulting frame
    if SHOW_REAL_VIDEO:
        cv2.imshow("Frame", frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
