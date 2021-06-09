import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pickle

def preProcess(x):    
    x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    x = cv.equalizeHist(x)
    x = x /255.0
    return x

def findClass(frame):
    frame = np.asarray(frame)
    frame = cv.resize(frame, (32,32))
    frame = preProcess(frame)
    
    frame = frame.reshape(1,32,32,1)
    return frame

pickle_in = open("number_trained.p","rb")
model = pickle.load(pickle_in)

listOfFile = os.listdir("testData")
s = " "

for i in listOfFile:
    img = cv.imread("testData/" + i)
    
    frame = findClass(img)
    classIndex = int(model.predict_classes(frame))
    predictions = model.predict(frame)
    probVal = np.amax(predictions)
    
    if probVal > 0.7:
      s = str(classIndex)  
    
    img = cv.resize(img, (300,300))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, thresh = cv.threshold(img_gray, 240, 255, cv.THRESH_BINARY)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)     

    contours,hierarchy=cv.findContours(opening.copy(),cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    
    center = None
    if len(contours)>0:
        c = max(contours, key = cv.contourArea)
        rect = cv.minAreaRect(c)        
        box = cv.boxPoints(rect)
        box = np.int64(box)              
        cv.drawContours(img, [box], 0, (255,0,0),2)
        cv.putText(img, s, (95,295), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
        
        cv.imshow("Resim", img)
        
        if (cv.waitKey(0) & 0xFF == 27) and (i.startswith("4")) == False:            
            continue
        else:
            break
        
cv.destroyAllWindows()