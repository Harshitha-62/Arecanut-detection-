#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install roboflow opencv-python-headless')


# In[11]:


import cv2
from roboflow import Roboflow
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Roboflow client
rf = Roboflow(api_key="")

# Load the model
project = rf.workspace().project("object-detection-l46bh")
model = project.version(1).model

# Access the camera
cap = cv2.VideoCapture(0)


def show_frame(frame):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

try:
    detected = False
    while not detected:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB (model expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference with a higher confidence threshold
        result = model.predict(rgb_frame, confidence=50, overlap=30).json()
        
        # Draw bounding boxes and labels on the frame
        for prediction in result['predictions']:
            print(f"Prediction: {prediction}")  # Debugging print
            if prediction['confidence'] >= 0.7 and prediction['class'].lower() == 'arecanut':
                x0, y0 = int(prediction['x'] - prediction['width'] / 2), int(prediction['y'] - prediction['height'] / 2)
                x1, y1 = int(prediction['x'] + prediction['width'] / 2), int(prediction['y'] + prediction['height'] / 2)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                label = f"{prediction['class']}: {prediction['confidence']:.2f}"
                cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detected = True
                break
        
        # Display the frame using matplotlib
        show_frame(frame)
        
except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    print("Capture released.")


# In[ ]:





# In[ ]:





# In[ ]:




