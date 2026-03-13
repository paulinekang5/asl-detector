import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# feature extractions
def extract_features(hand_landmarks):
    """
    media pipe returns 21 points
    
    - distances between key finger points
    - angles between joints
    - realtive hand orientation
    """
    
    if hand_landmarks is None:
        return None
    
    # Convert landmarks to a list of (x, y, z) tuples
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    features = []
    
   # captures the overall hand shape and finger positions
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            distance = np.linalg.norm(landmarks[i] - landmarks[j]) # euclidean distance
            features.append(distance)
    
    # normalize finger positions relative to the wrist (landmark 0)
    wrist = landmarks[0]
    for i in [4, 8, 12, 16, 20]:  # thumb, index, middle, ring, pinky tips
        relative_pos = landmarks[i] - wrist
        features.extend(relative_pos)
    
    # hand orientation (based on palm normal) -> distinguish between hands at different angles
    palm_normal = landmarks[9] - landmarks[0]  # vector from wrist to middle finger
    features.extend(palm_normal)
    
    return np.array(features)

