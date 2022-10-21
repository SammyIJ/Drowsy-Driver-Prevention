#Necessary Imports
from scipy.spatial import distance
from imutils import face_utils
from statistics import mean
from pygame import mixer
import imutils
import dlib
import cv2

threshold = 0.3  #The deafult EAR value below which an eye is considered closed, in case calibration fails
frameLimit = 30  #Number of frames eye can be closed before warning triggers

faceDetector = dlib.get_frontal_face_detector() #Initialize pre-trained face detector
shapePredictor = dlib.shape_predictor("facialLandmarkShapePredictor_68pt.dat") #Loads facial landmark predictor

#First and last points of the landmark shape predictor (LSP) that describe each feature
(L_eyeStart, L_eyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_eyeStart, R_eyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Initialize sounds
mixer.init()
alarm = mixer.Sound('beep.wav')
focus = mixer.Sound('focus16bit.wav')

face_utils.FACIAL_LANDMARKS_IDXS

#Input: eye_pts - a numpy array of 6 x,y co-ordinate pairs that make up the shape of the eye as per the LSP
#Returns: EAR   - a 64-bit float representing the eye's aspect ratio.
def calculate_EyeAspectRatio(eye_pts):
    #Vertical components
    X = distance.euclidean(eye_pts[1], eye_pts[5])
    Y = distance.euclidean(eye_pts[2], eye_pts[4])
    #Horizontal component
    Z = distance.euclidean(eye_pts[0], eye_pts[3])
    
    EAR = (X + Y) / (2.0 * Z)
    return EAR
def wakeDriver(frame):
    alarm.play()
    cv2.rectangle(frame,(0,0),(600,327),(0,0,255),20) #Red boarder
    cv2.putText(frame, "WAKE UP!", (225, 40),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "EMERGENCY", (210, 310),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
def alertDriver(frame, driverAlerted):
    if not driverAlerted:
        focus.play()
    cv2.putText(frame, "Face Not Detected...", (150, 165),cv2.FONT_HERSHEY_COMPLEX, 1, (100, 100, 200), 2)
    
    #Main
    
cam = cv2.VideoCapture(0) #Video Input w/ OpenCV
frameCount = 0            #Tracks how long eyes have been closed
driving = True
speed = 72                #Current Vehicle speed
minimumSpeed = 30         #Speed at which the Fatigue Prevention system activates
undetectedDuration = 0    #Monitors how long a face has gone undetected for
driverAlerted = False

#The variables below are concerned with calculating user-specific threshold EAR values at which alarm is triggered
calibrationEARs = []        #List containing first user EAR values to be averaged
currentFrame = 0            #Frame counter, for time keeping
calibrationDuration = 40    #Duration (frames) of the calibration period
thresholdPercentage = 0.75  #Percentage of resting EAR to set threshold at
thresholdCalculated = False 

while driving and speed > minimumSpeed:
    #Exctract frame + preprocess
    _, frame = cam.read() #Grabs the current frame of the camera feed
    frame = imutils.resize(frame, width=600)       #Resizes
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Greyscale
    
    #Detect faces within grescale image
    faces = faceDetector(grey,1)
    
    #Triggers one-time gentle Alert if no face detected for short period of time
    if len(faces) == 0:
        undetectedDuration += 1
        if undetectedDuration > frameLimit:
            alertDriver(frame, driverAlerted)
            driverAlerted = True
    else:
        undetectedDuration = 0
        driverAlerted = False
    
    
    for face in faces:
        #Determine facial landmarks for face ROI, then convert co-ords to NumPy array
        shape = shapePredictor(grey, face)
        shape = face_utils.shape_to_np(shape)
        
        #Draw bounding box around detected face
        (x, y, w, h) = face_utils.rect_to_bb(face) #convert dlib rect to openCV bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100),1)
        
        cv2.putText(frame,"Driver", (x-5, y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,100),1)
        
        #Define set of points (from LSP) that constitutes each feature
        left_eye = shape[L_eyeStart:L_eyeEnd]
        right_eye = shape[R_eyeStart:R_eyeEnd]
        
        #Calculate Eye Aspect Ratio for each eye
        left_EAR = calculate_EyeAspectRatio(left_eye)
        right_EAR = calculate_EyeAspectRatio(right_eye)
        EAR = (left_EAR + right_EAR) / 2.0  #Take Average
        
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (15,310),cv2.FONT_HERSHEY_SIMPLEX,0.85, (100,100,100),2)
        
        try:
            #Calibrate EAR threshold values to user
            if currentFrame < calibrationDuration:
                cv2.putText(frame, "Calibrating...", (15,280),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255),1)
                calibrationEARs.append(EAR)
            elif thresholdCalculated == False:
                threshold = mean(calibrationEARs) * thresholdPercentage
                cv2.putText(frame, "Threshold: {:.2f}".format(threshold), (15,280),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,200),1)
                thresholdCalculated == True
        except:
            cv2.putText(frame, "Face unable to be detected upon launch - using default EAR threshold", 
                        (15,280),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,255),1)
        
        #Use OpenCV to draw eye contours
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        
        #Detects when EAR falls below calculated threshold
        if EAR < threshold:
            frameCount += 1
            #print(frameCount)
            if frameCount >= frameLimit:
                #If the driver is asleep, we wake them up
                wakeDriver(frame)
        else:
            frameCount = 0; #reset alarm timer when eyes open again
            
    currentFrame += 1
        
    #Display the camera feed, press x to close program
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        cv2.destroyAllWindows()
        cam.release()
        break
        
        
        
# Please note: 'NoneType' object has no attribute 'shape' error message indicates issue with the webcam,
#               ensure it is detectable by the computer and fully functioning.