import cv2
import os
import sys
import numpy as np
import time
import json

########################################################################
# SETTINGS #
# Set this to True to get real camera video from cv2
SHOW_REAL_VIDEO = ("--show" in sys.argv[:])

# Save images to output folder
SAVE_IMAGES = ("-s" in sys.argv[:] or "--save" in sys.argv[:])

# Log to a json file
LOG_TIMES = ("-l" in sys.argv[:] or "--log" in sys.argv[:])

# Check for option for first frame
SHOW_START_FRAME = ("-i" in sys.argv[:] or "--initial" in sys.argv[:])


# Max pixel difference over which to trigger the motion alarm
THRESHOLD = 100
# sidewalk, street, or both
AREA = "sidewalk"

########################################################################


def format_frame(inputFrame, scanArea):
    # Convert to grayscale
    gray = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

    # Return cropped frame
    # Sidewalk
    if scanArea == "sidewalk":
        return gray[480:-150, 450:-350]

    # Street (No Parked Car)
    elif scanArea == "street":
        return gray[0:-650, 455:-560], inputFrame[0:-325, 225:-280]

    # Street and sidewalk (NO TREES)
    else:
        return gray[0:-150, 450:-560], inputFrame[0:-75, 225:-280]


def format_save(inputFrame, scanArea):
    # Return cropped frame
    # Sidewalk
    if scanArea == "sidewalk":
        return inputFrame[240:-75, 225:-125]

    # Street (No Parked Car)
    elif scanArea == "street":
        return inputFrame[0:-325, 225:-280]

    # Streen and sidewalk (NO TREES)
    else:
        return inputFrame[0:-75, 225:-280]


def saveImage(inputFrame, fileNamePrefix):
    fullFileName = "outputs/" + fileNamePrefix + ".png"
    cv2.imwrite(fullFileName, inputFrame)
    print("SAVED IMAGE FINISHED")


# Open json log file if log option
with open("log.json") as file:
    JSON_RECORD = json.load(file)


# START OF MAIN #
cap = cv2.VideoCapture(0)
time.sleep(1)
start_frame = format_frame(cap.read()[1], AREA)
print("First Frame:")
print(start_frame)

if SHOW_START_FRAME:
    cv2.imshow("Start Frame", start_frame)


SUM_FRAME = start_frame.astype(np.uint32)
AVG_FRAME = start_frame.astype(np.uint32)

# For saving frames
records = []
frame_buffer = []
start = None
mostRecent = None

GAP = 10

counter = 1

while(cv2.waitKey(1) & 0xFF != ord('q')):
    # Get image data
    full_frame = cap.read()[1]
    
    frame = format_frame(full_frame, AREA)
    saveFrame = format_save(full_frame, AREA)

    DIFF = cv2.absdiff(np.float32(AVG_FRAME), np.float32(frame))
    MAX_DIFF = np.amax(DIFF)

    print(counter)
    print(f"MAX_DIFF: {MAX_DIFF}\n")

    # Check for change dist from avg
    if MAX_DIFF > THRESHOLD:
        print("\t\t\tMOVEMENT DETECTED!")

        if not start:
            start = counter
            start_time = int(time.time())
        mostRecent = counter
        frame_buffer.append(saveFrame)

    else:
        if counter - GAP == mostRecent and (counter - start) > GAP * 2 and len(frame_buffer) > GAP * 2: # ADD CHECK FOR LENGTH OF BUFFER FOR LEGIT MOVEMENT
            entry = {"start_frame":start, "finish_frame":counter - GAP, "start_time":start_time, "finish_time":int(time.time()), "window_length":len(frame_buffer)}
            records.append(entry)
            frame_selection = int(entry["window_length"] / 2)

            if SAVE_IMAGES:
                print(f"length of buffer: {len(frame_buffer)}")
                print(f"saving image with saveImage(frame_buffer[frame_selection = {frame_selection}], str(entry['start_time' = {str(entry['start_time'])}])")
                saveImage(frame_buffer[frame_selection], str(entry["start_time"]))
            
            if LOG_TIMES:
                JSON_RECORD.append(entry["start_time"])
                with open("log.json", 'w') as file:
                    json.dump(JSON_RECORD, file)

            frame_buffer = []
            start = None
            mostRecent = None

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
