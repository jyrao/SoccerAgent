import os
import face_recognition
import numpy as np
import pickle

PROJECT_PATH = "PROJECT_PATH" # Replace with actual project path

def build_face_library(base_path): # Base path to the SoccerWiki directory containing players' images.
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    face_library = {}
    person_folders = sorted([folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))])

    for person_folder in person_folders:
        person_folder_path = os.path.join(base_path, person_folder)
        
        img_files = sorted([file for file in os.listdir(person_folder_path) if os.path.splitext(file)[1].lower() in VALID_IMAGE_EXTENSIONS])

        for img_file in img_files:
            img_path = os.path.join(person_folder_path, img_file)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]

                face_library[person_folder] = face_encoding
                print(f"Added {person_folder} to the face library.")
                break

    with open("face_library.pkl", 'wb') as f:
        pickle.dump(face_library, f)
    print(f"Face library saved to face_library.pkl.")

    return face_library


def FACE_RECOGNITION(query=None, material=[]):
    filename = f"{PROJECT_PATH}/pipeline/toolbox/utils/face_library.pkl" # Replace with actual path to the face library
    with open(filename, 'rb') as f:
        face_library = pickle.load(f)

    new_img_path = material[0]
    new_image = face_recognition.load_image_file(new_img_path)
    new_face_encodings = face_recognition.face_encodings(new_image)
    
    if new_face_encodings:
        new_face_encoding = new_face_encodings[0]

        distances = []
        for person_name, stored_encoding in face_library.items():
            distance = face_recognition.face_distance([stored_encoding], new_face_encoding)
            distances.append((person_name, distance[0]))
        
        distances.sort(key=lambda x: x[1])
        most_similar_person = distances[0]
        # print(f"Match found: {most_similar_person[0]} with distance: {most_similar_person[1]}")
        
        return f"The person in the photo is most likely: {most_similar_person[0]}, distance: {most_similar_person[1]}"
    
    else:
        # print("No face detected in the new image.")
        return "None"
