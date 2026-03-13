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

# data collection
def collect_training_data(letter, num_samples=50):
    """
    collect training data for a specific ASL letter:  user performs the ASL letter in front of the camera,
    and we save the extracted features to our training dataset.
    
    returns: list of feature vectors for this letter
    """
    
    print(f"\n{'='*60}")
    print(f"Collecting data for letter: {letter}")
    print(f"Press SPACE to start collecting {num_samples} samples")
    print(f"Press 'q' to skip this letter")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    collected_features = []
    samples_collected = 0
    collecting = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # flip the frame for a selfie-view -> mirror camera 
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # convert bgr to rgb for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        # display status
        status_text = "COLLECTING..." if collecting else "READY (press SPACE)"
        color = (0, 255, 0) if collecting else (0, 0, 255)
        cv2.putText(frame, f"Letter: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # drawing landmarks/points on the video feed
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # collecting: extract
            if collecting and result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                features = extract_features(hand_landmarks)
                if features is not None:
                    collected_features.append(features)
                    samples_collected += 1
                    
                    
                    if samples_collected >= num_samples:
                        print(f"Finished collecting {num_samples} samples for letter {letter}!")
                        break
        
        cv2.imshow(f"Collecting ASL Letter: {letter}", frame)
        
        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # space = stop
            collecting = not collecting
            if collecting:
                print(f"Started collecting for {letter}")
        elif key == ord('q'):  # q = skip
            print(f"Skipped letter {letter}")
            break
        elif key == 27:  # esc = exit
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    return collected_features


def create_training_dataset(num_samples_per_letter=50):
    """
    make training dataset by collecting data for all ASL letters.
    
    - loops through each letter and calls collect_training_data.
    - result: dictionary with all collected feature vectors organized by letter.
    """
    
    training_data = {}
    
    for letter in ASL_LETTERS:
        features_list = collect_training_data(letter, num_samples_per_letter)
        
        if features_list is None:
            # User pressed ESC to exit
            print("Data collection cancelled by user")
            return training_data if training_data else None
        
        if features_list:
            training_data[letter] = features_list
            print(f"Saved {len(features_list)} samples for letter {letter}")
    
    return training_data


def save_training_data(training_data, filename=DATA_FILE):
    """
    save collected data
    """
    
    with open(filename, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Training data saved to {filename}")


def load_training_data(filename=DATA_FILE):
    """
    load previously saved training data
    """
    
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded training data from {filename}")
    return data
