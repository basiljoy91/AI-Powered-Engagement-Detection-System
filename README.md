# AI-Powered Engagement Detection System

An AI-based system that detects user engagement levels in real-time using **computer vision** and **machine learning**. The model analyzes **body posture** through **OpenCV** and **MediaPipe**, classifies engagement as **Engaged, Distracted, or Needs Help**, and provides **automated verbal assistance** using **text-to-speech (TTS)**. This project demonstrates **real-time engagement tracking** and **adaptive learning support**.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project

This **AI-Powered Engagement Detection System** is designed to **analyze user engagement** in real-time using **computer vision** and **machine learning**. The system processes the **user's posture** from the webcam feed to classify their engagement level as **Engaged, Distracted, or Needs Help**. Based on the analysis, it provides **verbal assistance** to help users stay focused.

The system is implemented using **OpenCV**, **MediaPipe**, and **Scikit-learn**, and uses **text-to-speech (TTS)** to give feedback. The model improves over time, making the system **adaptive** to different users' behaviors.

---

## Technologies Used

- **Python**: The programming language used for developing the system.
- **OpenCV**: For real-time image processing and capturing video.
- **MediaPipe**: For pose detection and extracting body landmarks.
- **Scikit-learn**: For machine learning model training (Random Forest Classifier).
- **pyttsx3**: For **Text-to-Speech (TTS)** interaction.
- **SpeechRecognition**: For voice command processing (if extended).
  
---

## Features

- **Real-time Engagement Detection**: Detects whether the user is **Engaged, Distracted**, or **Needs Help** using body posture analysis.
- **Automated Feedback**: Provides **verbal assistance** through **Text-to-Speech (TTS)** if the user is struggling or distracted.
- **Adaptive Learning**: The system improves over time with **machine learning** and adapts based on users’ interactions.
- **Real-time Processing**: Uses the **webcam** to analyze engagement in real-time and provide instant feedback.
  
---
