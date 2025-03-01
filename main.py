import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import speech_recognition as sr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Initialize modules
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Sample Data Collection Function (Simulated for Demo)
def collect_data():
    X = np.random.rand(100, 5)  # Simulated 5 posture-based features
    y = np.random.choice(['Engaged', 'Distracted', 'Needs Help'], 100)
    return X, y

# Train ML Model
def train_model():
    X, y = collect_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model Trained!")
    return model, scaler

# Robot Speech Response
def robot_response(message):
    print(f"System: {message}")
    engine.say(message)
    engine.runAndWait()

# Live Engagement Detection & Automated Response
def detect_engagement():
    model, scaler = train_model()
    cap = cv2.VideoCapture(0)
    last_response_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            posture_features = np.array([landmarks[i].y for i in [11, 12, 23, 24, 0]])
            posture_features = scaler.transform([posture_features])
            prediction = model.predict(posture_features)[0]
            
            cv2.putText(frame, f"Engagement: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Trigger voice response every 5 seconds if needed
            if prediction == "Needs Help" and (time.time() - last_response_time) > 5:
                robot_response("I see you're struggling. Let me know if you need assistance.")
                last_response_time = time.time()
        
        cv2.imshow("Engagement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_engagement()
