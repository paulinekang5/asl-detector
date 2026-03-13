
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


def train_random_forest_model(training_data, test_size=0.2, random_state=42):
    """
    why random forest:
    - works well with the feature vectors 
    - fast to train and for inference
    - handles non-linear relationships between features
    - interpretable (can see which features are important)
    """
    
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # prepare data
    # convert dictionary into X (features) and y (labels)
    X = []  # feature vectors
    y = []  # labels (letters)
    
    for letter, features_list in training_data.items():
        for features in features_list:
            X.append(features)
            y.append(letter)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # split into training and testin
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y 
    )
    
    
    # n_estimators = number of decision trees in  forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20, 
        min_samples_split=5,  
        random_state=random_state,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.2%}")
    print(f"  Testing Accuracy:  {test_accuracy:.2%}")
    
    print("\nAccuracy per letter:")
    for letter in sorted(ASL_LETTERS):
        mask = y_test == letter
        if mask.sum() > 0:
            letter_accuracy = (y_test_pred[mask] == letter).sum() / mask.sum()
            print(f"  {letter}: {letter_accuracy:.2%}")
    
    # create label encoder dictionary for later use
    label_encoder = {letter: i for i, letter in enumerate(ASL_LETTERS)}
    
    return model, ASL_LETTERS, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }


def save_model(model, label_encoder, model_file=MODEL_FILE, label_file=LABEL_ENCODER_FILE):
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(label_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Model saved to {model_file}")
    print(f"Labels saved to {label_file}")


def load_model(model_file=MODEL_FILE, label_file=LABEL_ENCODER_FILE):
    if not os.path.exists(model_file) or not os.path.exists(label_file):
        return None, None
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(label_file, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded from {model_file}")
    return model, label_encoder