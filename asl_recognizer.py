import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json

import data_collection 
import feature_extraction 
import model_training
import real_time_recognizer
import setup


def main_menu():
    print("\n" + "="*60)
    print("ASL ALPHABET RECOGNIZER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run real-time recognition")
        print("4. Collect data + Train + Run (full pipeline)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # collect training data
            num_samples = input("How many samples per letter? (default 50): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 50
            
            training_data = data_collection.create_training_dataset(num_samples)
            if training_data:
                data_collection.save_training_data(training_data)
        
        elif choice == '2':
            # train model
            training_data = data_collection.load_training_data()
            if training_data is None:
                print("No training data found. Collect data first (option 1)")
                continue
            
            model, label_encoder, accuracy = model_training.train_random_forest_model(training_data)
            data_collection.save_model(model, label_encoder)
        
        elif choice == '3':
            # run real time recognition
            model, label_encoder = model_training.load_model()
            if model is None:
                print("No model found. Train a model first (option 2)")
                continue
            
            real_time_recognizer.run_real_time_recognition(model, label_encoder)
        
        elif choice == '4':
            # everythin
            num_samples = input("How many samples per letter? (default 50): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 50
            
            print("\nStep 1: Collecting training data...")
            training_data = data_collection.create_training_dataset(num_samples)
            if training_data:
                data_collection.save_training_data(training_data)
                
                print("\nStep 2: Training model...")
                model, label_encoder, accuracy = model_training.train_random_forest_model(training_data)
                model_training.save_model(model, label_encoder)
                
                print("\nStep 3: Running real-time recognition...")
                real_time_recognizer.run_real_time_recognition(model, label_encoder)
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()