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
    # Simulating 100 data points with 5 posture-based features
    X = np.random.rand(100, 5)  # Simulated posture features
    y = np.random.choice(['Engaged', 'Distracted', 'Needs Help'], 100)  # Labels
    return X, y

# Train ML Model
def train_model():
    X, y = collect_data()  # Collect simulated data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()  # Standardizing the data
    X_train = scaler.fit_transform(X_train)
    model = RandomForestClassifier()  # Random Forest classifier
    model.fit(X_train, y_train)
    print("Model Trained!")
    return model, scaler

# Robot Speech Response
def robot_response(message):
    print(f"System: {message}")
    engine.say(message)
    engine.runAndWait()

# Posture feature extraction function (simplified example)
def extract_posture_features(landmarks):
    # We take a simplified approach here by extracting 5 key features
    # For a real model, you should calculate angles and more sophisticated features
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    hip_left = landmarks[23]
    hip_right = landmarks[24]
    nose = landmarks[0]
    
    # Example: Calculate the distances between specific body parts
    shoulder_distance = np.linalg.norm(np.array([shoulder_left.x, shoulder_left.y]) - np.array([shoulder_right.x, shoulder_right.y]))
    hip_distance = np.linalg.norm(np.array([hip_left.x, hip_left.y]) - np.array([hip_right.x, hip_right.y]))
    nose_height = nose.y  # Simplified height feature
    
    return np.array([shoulder_distance, hip_distance, nose_height])

# Live Engagement Detection & Automated Response
def detect_engagement():
    model, scaler = train_model()
    cap = cv2.VideoCapture(0)  # Start the webcam
    last_response_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            posture_features = extract_posture_features(landmarks)  # Extract posture features
            
            posture_features = scaler.transform([posture_features])  # Scale features
            prediction = model.predict(posture_features)[0]
            
            # Display the engagement prediction on the frame
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
