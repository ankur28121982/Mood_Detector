# Project Title
DeepFace Mood Detector

## Project Description
This project utilizes the DeepFace library to detect emotions from a live video feed using a webcam. The detected emotions are logged to a CSV file for tracking mood swings over time. Additionally, the detected face is highlighted with a rectangle, and the emotion is displayed on the screen.

## Quick Steps to Run the Project:
1. Save the Python file (e.g., `mooddetec.py`) in a directory/folder.
2. Ensure that the required packages are installed (see Requirements).
3. Run the script to start the mood detection.

## Requirements
- Python 3.8
- OpenCV
- DeepFace
- Pandas
- TensorFlow 1.14.0
- Numpy 1.16.4

## Installation
To set up the environment and install the required packages, follow these steps:

1. Clone the repository:

git clone https://github.com/ankur28121982/Mood_Detector

2. Navigate to the project directory:

cd deepface-mood-detector


3. Create and activate a virtual environment:


conda create -n deepface_env python=3.8
conda activate deepface_env

4. Install the required packages:

pip install numpy==1.16.4
pip install tensorflow==1.14.0
pip install opencv-python-headless
pip install deepface
pip install pandas
```

## Problems Faced and Solutions

### Problem: Incompatible `numpy` Version
Initially, there was an issue with `numpy` versions. The installed version was not compatible with TensorFlow and DeepFace, causing errors. 

#### Solution:
We created a new conda environment with the correct version of `numpy`:

conda create -n deepface_env python=3.8 numpy=1.16.4
conda activate deepface_env
pip install tensorflow==1.14.0
pip install opencv-python-headless
pip install deepface
pip install pandas
```

### Problem: Face Detection Error
An error was encountered where the face could not be detected in the numpy array. The solution was to set `enforce_detection` to `False` in the DeepFace analyze method.

### Problem: List Indices Error
An error `list indices must be integers or slices, not str` was encountered. This was due to the incorrect handling of the result from DeepFace.

#### Solution:
Modify the result handling to access the first element of the result list:

```python
emotion = result[0]['dominant_emotion']
```

### Problem: Webcam Not Capturing
At times, the webcam failed to capture images correctly.

#### Solution:
Ensure that the webcam is properly connected and accessible. Use OpenCV to capture video feed.

## Running the Script

To run the script, use the following command:

python mooddetec.py

## Script Overview

### `mooddetec.py`

import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime

# Function to detect emotions using DeepFace
def detect_emotion(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Error analyzing emotion: {str(e)}")
        return None

# Function to save log
def save_log(emotion):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {'Time': current_time, 'Emotion': emotion}
    df = pd.DataFrame([log_entry])
    with open('deepface.csv', 'a') as f:
        df.to_csv(f, header=f.tell()==0, index=False)

# Main function to capture video feed and detect emotions
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Flip the frame horizontally for better view
        frame = cv2.flip(frame, 1)
        
        # Detect emotion
        emotion = detect_emotion(frame)
        if emotion:
            print(f"Detected Emotion: {emotion}")
            save_log(emotion)
        
        # Draw a rectangle around the face and display the emotion on the screen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display the frame
        cv2.imshow('Emotion Detector', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Author
Dr. Ankur Chaturvedi
ankur1122@gmail.com

Dr. Ankur Chaturvedi is a seasoned Transformation Specialist with expertise in Consulting, Data Science, and Quality Management. He has a profound background in LEAN and Agile Transformation, having managed and optimized processes for teams of up to 3000 FTEs. Dr. Chaturvedi is currently a Senior Manager at Infosys BPM, where he spearheads process excellence, quality consulting, and organizational improvements. His skill set includes deploying data analytics, robotics, and mindset & behavior tools to drive efficiency and transformation across various domains.
