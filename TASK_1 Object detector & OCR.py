#!/usr/bin/env python
# coding: utf-8

# ## TSF - GRADUATE ROTATIONAL INTERNSHIP PROGRAM (GRIP)
# 
# ## COMPUTER VISION & INTERNET OF THINGS
# 
# ## TASK 1 - OBJECT DETECTION / OPTICAL CHARACTER RECOGNITION (OCR)
# 
# ## AUTHOR - ADITHYA BASKER
# 
# ### #GRIPJAN22
# 
# ### 1. OBJECTION DETECTION

# In[1]:


import cv2

# Source data
img_file = "C:/Users/Adithya Basker/Downloads/img1.png"

# create an openCV image
img = cv2.imread(img_file)

# pre trained Car, Pedestrian, bus, and two-wheeler classifiers
car_classifier = 'cars.xml'
pedestrian_classifier = 'pedestrian.xml'
bus_classifier = 'Bus_front.xml'
twowheeler_classifier = 'two_wheeler.xml'

# convert color image to grey image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create trackers using classifiers using OpenCV
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)
bus_tracker = cv2.CascadeClassifier(bus_classifier)
twowheeler_tracker = cv2.CascadeClassifier(twowheeler_classifier)

# detect objects
cars = car_tracker.detectMultiScale(gray_img)
pedestrian = pedestrian_tracker.detectMultiScale(gray_img)
bus = bus_tracker.detectMultiScale(gray_img)
twowheeler = twowheeler_tracker.detectMultiScale(gray_img)

# display the coordinates of different objects - multi dimensional array
print(cars)
print(pedestrian)
print(bus)
print(twowheeler)

# draw rectangle around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# draw rectangle around the pedestrian
for (x,y,w,h) in pedestrian:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.putText(img, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# draw rectangle around the bus
for (x,y,w,h) in bus:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.putText(img, 'Bus', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# draw rectangle around the two wheeler
for (x,y,w,h) in twowheeler:
    cv2.rectangle(img, (x,y), (x+w, y+h), (216,255,0), 2)
    cv2.putText(img, 'Two Wheeler', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Finally display the image with the markings
cv2.imshow('THE DETECTED OBJECTS',img)

# wait for the keystroke to exit
cv2.waitKey()


print("DONE!")


# ### 2. OPTICAL CHARACTER RECOGNITION (OCR)

# In[2]:


import cv2
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd=r'C:/Users/Adithya Basker/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
img = cv2.imread("testimgOCR.jpeg")
img = cv2.resize(img, (400, 450))
cv2.imshow("Image", img)
text = pytesseract.image_to_string(img)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


imgboxes = pytesseract.image_to_boxes(img)
print(imgboxes)


# In[4]:


imgh,imgw,_ = img.shape
for boxes in imgboxes.splitlines():#make bonding boxes 
    boxes = boxes.split(" ")
    x,y,w,h = int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
    cv2.rectangle(img,(x,imgh-y),(w,imgh-h),(0,0,255),3 )
    cv2.putText(img,boxes[0],(x,imgh-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
plt.imshow(img)#by default cv2 is BGR


# In[7]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[ ]:




