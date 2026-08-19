import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

def create_embeddings(image_folder):
    # Initialize face analysis model
    detector = FaceAnalysis(name='buffalo_l')  # Use 'buffalo_l' or other models as needed
    detector.prepare(ctx_id=-1)

    embeddings = []  # List to store embeddings
    names = []  # List to store corresponding names (based on filenames)

    # Iterate through images in the specified folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Check for image file types
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)

            # Perform face detection
            faces = detector.get(img)

            if faces is not None and len(faces) > 0:
                # Assuming we only want the first detected face
                face_embedding = faces[0].normed_embedding
                embeddings.append(face_embedding)
                names.append(filename.split('.')[0])  # Use the file name (without extension) as the name

    # Convert embeddings to a NumPy array for consistency
    embeddings = np.array(embeddings)

    # Save embeddings and names to a .pkl file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings, names), f)

    print(f"Embeddings saved: {embeddings.shape} for {len(names)} images.")

if __name__ == '__main__':
    image_folder = 'static'  # Replace with your image folder path
    create_embeddings(image_folder)

