
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json

from data_collection import *
from feature_extraction import *
from model_trianing import *
from real_time_reconizer import *
from setup import *


def run_real_time_recognition(model, label_encoder):
    """
    run real-time asl letter recognition on webcam feed.
    
    what it does:
    1. captures frames from webcam
    2. detects hand landmarks with mediapipe
    3. extracts features
    4. runs the trained model to predict letter
    5. displays the predicted letter on screen
    """
    
    print("\n" + "="*60)
    print("REAL-TIME ASL LETTER RECOGNITION")
    print("Press 'q' to quit")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    
    # variables to smooth predictions (reduce jitter from movement)
    prediction_history = []
    history_size = 5  # use last 5 predictions to determine final output
    current_letter = ""
    confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # convert to rgb for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        # hand detected = make a prediction
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            
            # draw features on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # extract features
            features = extract_features(hand_landmarks)
            
            if features is not None:
                # make prediction
                prediction = model.predict([features])[0]
                confidence_score = model.predict_proba([features]).max()
                
                # add to history
                prediction_history.append((prediction, confidence_score))
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # use majority vote from history
                if len(prediction_history) > 0:
                    predictions = [p[0] for p in prediction_history]
                    most_common = max(set(predictions), key=predictions.count)
                    confidence = np.mean([p[1] for p in prediction_history if p[0] == most_common])
                    
                    if confidence > 0.6:  # only show if confident!
                        current_letter = most_common
        else:
            prediction_history = []
            current_letter = ""
            confidence = 0.0
        
        # display 
        cv2.putText(frame, f"Letter: {current_letter}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.imshow("ASL Letter Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
