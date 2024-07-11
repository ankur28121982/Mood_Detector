import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime

# Function to detect emotions using DeepFace
def detect_emotion(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return result
    except Exception as e:
        print(f"Error analyzing emotion: {str(e)}")
        return None

# Function to save mood log
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
        result = detect_emotion(frame)
        if result:
            if isinstance(result, list):
                result = result[0]  # Assuming the list contains one dictionary
                
            emotion = result.get('dominant_emotion')
            region = result.get('region')
            
            if region:
                # Draw a rectangle around the face
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Display the emotion on the frame
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                print(f"Detected Emotion: {emotion}")
                save_log(emotion)
        
        # Display the frame
        cv2.imshow('Emotion Detector', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
