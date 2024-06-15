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



def extract_faces_from_video(video_path, output_dir):
    # Apri il video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore nell'apertura del video: {video_path}")
        return

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Inizializza i conteggi per le persone identificate
    person_counter = 1
    person_bookings = {}  # Dizionario per tenere traccia delle immagini per ogni persona

    # Processa il video fotogramma per fotogramma
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Frame {frame_count}")

        # Rileva i volti nel fotogramma corrente
        face_locations = face_recognition.face_locations(frame)
        
        # Estrai e salva i volti trovati
        for idx, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location

            # Ritaglia e salva il volto trovato
            face_image = frame[top:bottom, left:right]
            person_id = f"person_{person_counter}"
            person_output_dir = os.path.join(output_dir, person_id)
            os.makedirs(person_output_dir, exist_ok=True)
            output_path = os.path.join(person_output_dir, f"{person_id}_{frame_count}_{idx}.jpg")
            cv2.imwrite(output_path, face_image)

            # Aggiungi l'immagine alla prenotazione della persona
            if person_id not in person_bookings:
                person_bookings[person_id] = []
            person_bookings[person_id].append(output_path)

            print(f"Salvato volto di {person_id} in {output_path}")

            # Incrementa il contatore delle persone
            person_counter += 1

    # Chiudi il video
    cap.release()

    print("Prenotazioni fotografiche completate.")
    return person_bookings

# Esempio di utilizzo
if __name__ == "__main__":
    video_path = 'test.mp4'  # Inserisci il percorso del tuo video
    output_dir = 'output_booking'  # Cartella di output per le prenotazioni fotografiche

    # Estrai i volti dal video e crea le prenotazioni fotografiche
    person_bookings = extract_faces_from_video(video_path, output_dir)

    # Stampa le prenotazioni fotografiche
    print("\nPrenotazioni fotografiche:")
    for person_id, images in person_bookings.items():
        print(f"{person_id}:")
        for image_path in images:
            print(f" - {image_path}")