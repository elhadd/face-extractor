import os
import cv2
import argparse
import filetype as ft
import numpy as np
from pathlib import Path
from PIL import Image
from facedetector import FaceDetector
import face_recognition
from collections import defaultdict  # Importa defaultdict da collections
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
import json
import shutil
import tensorflow as tf
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import dlib
from deepface import DeepFace

        
def confronta_volto_con_face_recognition(file_immagine1, file_immagine2):

    # Carica le immagini e ottieni i vettori di encoding dei volti
    immagine1 = face_recognition.load_image_file(file_immagine1)
    immagine2 = face_recognition.load_image_file(file_immagine2)
    
    # Trova i volti nelle immagini
    volti1 = face_recognition.face_locations(immagine1)
    volti2 = face_recognition.face_locations(immagine2)
    
    if len(volti1) == 0 or len(volti2) == 0:
        raise Exception("Nessun volto rilevato nelle immagini.")
        
    if len(volti1) != 1:
        raise Exception(f"Numero errato di volti rilevati nell'immagine 1: {len(volti1)}")
    
    if len(volti2) != 1:
        raise Exception(f"Numero errato di volti rilevati nell'immagine 2: {len(volti2)}")
    
    # Genera i vettori di encoding dei volti
    encoding1 = face_recognition.face_encodings(immagine1, volti1)[0]
    encoding2 = face_recognition.face_encodings(immagine2, volti2)[0]
    
    # Calcola la distanza euclidea tra i vettori di encoding
    distance = face_recognition.face_distance([encoding1], encoding2)
    
    # Converto la distanza in un punteggio di similitudine (quanto più vicino a 1 è migliore)
    similarity_score = 1 - distance[0]
    
    # Converto il punteggio in percentuale
    similarity_percent = similarity_score * 100
    
    print(f"Percentuale di similarità tra i volti con Face Recognition: {similarity_percent:.2f}%")



def confronta_volto_deepface(file_immagine1, file_immagine2):
    try:
        # Verifica che i file delle immagini esistano
        if not os.path.isfile(file_immagine1):
            raise ValueError(f"File non trovato: {file_immagine1}")
        if not os.path.isfile(file_immagine2):
            raise ValueError(f"File non trovato: {file_immagine2}")

        # Confronta i volti e ottieni la distanza utilizzando RetinaFace come detector_backend
        result = DeepFace.verify(img1_path=file_immagine1, img2_path=file_immagine2, detector_backend='retinaface', model_name='Facenet512')

        # Calcola la similarità come complemento della distanza
        similarity_score = 1 - result['distance']
        similarity_percent = similarity_score * 100

        # Stampare la percentuale di similarità formattata
        print(f"Percentuale di similarità tra i volti con DeepFace usando RetinaFace: {similarity_percent:.2f}%")

    except ValueError as ve:
        print(f"Errore: {str(ve)}")
    except Exception as e:
        print(f"Errore durante il confronto dei volti con DeepFace: {str(e)}")
        
    

def main(args):
    image1_name = args["image1"]
    image2_name = args["image2"]
    
    # Costruisci il percorso completo per l'immagine 1 considerando la sottocartella 'output'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    image1_path = os.path.join(output_dir, image1_name)
    
    # Costruisci il percorso completo per l'immagine 2 considerando la sottocartella 'output'
    image2_path = os.path.join(output_dir, image2_name)
    
    confronta_volto_con_face_recognition(image1_path,image2_path)
    confronta_volto_deepface(image1_path,image2_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # options
    parser.add_argument("-a", "--image1", required=True, help="Nome dell'immagine 1")
    parser.add_argument("-b", "--image2", required=True, help="Nome dell'immagine 2")
    
    args = vars(parser.parse_args())
    main(args)