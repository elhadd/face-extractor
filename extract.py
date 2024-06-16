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


def rename_files_from_duplicates(duplicates_json_path, folder_path):
    # Carica il file JSON contenente i gruppi di duplicati
    with open(duplicates_json_path, 'r') as f:
        duplicates = json.load(f)
    
    # Contatore per numerare i file rinominati
    count = 1

    # Itera attraverso ogni gruppo di duplicati nel dizionario
    for key, value_list in duplicates.items():
        # Genera un nuovo nome per il gruppo di duplicati
        new_name_prefix = f"Soggetto_{count}"
        count += 1
        
        # Rinomina tutti i file nel gruppo di duplicati
        for filename in [key] + value_list:
            old_path = os.path.join(folder_path, filename)
            new_filename = f"{new_name_prefix}_{count:02d}.jpg"  # Aggiungi l'estensione appropriata se necessario
            new_path = os.path.join(folder_path, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"File rinominato: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Errore durante la rinomina di {filename}: {e}")

    print("Rinomina dei file completata.")


def calculate_face_visibility(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    total_face_area = 0
    for (x, y, w, h) in faces:
        total_face_area += w * h
    
    return total_face_area

def calculate_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance

def get_best_frames_per_second(video_path):
    video = cv2.VideoCapture(video_path)
    frames_per_second = []
    
    frame_count = 0
    while True:
        # Imposta la posizione del video al frame del secondo successivo
        video.set(cv2.CAP_PROP_POS_MSEC, (frame_count * 1000))
        success, frame = video.read()

        if not success:
            break
        
        face_visibility = calculate_face_visibility(frame)
        image_quality = calculate_image_quality(frame)
        
        frames_per_second.append((frame_count, frame, face_visibility, image_quality))
        frame_count += 1
    
    video.release()
    cv2.destroyAllWindows()
    
    # Filtra i frame per mantenere solo il migliore per ogni secondo
    best_frames = {}
    for frame_count, frame, face_visibility, image_quality in frames_per_second:
        if frame_count not in best_frames or (face_visibility > best_frames[frame_count][1] or (face_visibility == best_frames[frame_count][1] and image_quality > best_frames[frame_count][2])):
            best_frames[frame_count] = (frame, face_visibility, image_quality)
    
    return [best_frames[second][0] for second in sorted(best_frames)]

def check_faces(files, inputDir, outputDir, padding):
    images = []
    
    for file in files:
        dir, path, mime, filename = file.values()
        targetDir = os.path.join(outputDir, os.path.relpath(dir, inputDir))
        
        if mime is None:
            continue
        
        if mime.startswith('video'):
            print(f'[INFO] Extracting frames from video: {filename}')
            best_frames = get_best_frames_per_second(path)
            
            for frame in best_frames:
                image = {
                    "file": frame,
                    "sourcePath": path,
                    "sourceType": "video",
                    "targetDir": targetDir,
                    "filename": filename
                }
                images.append(image)
        
        elif mime.startswith('image'):
            image = {
                "file": cv2.imread(path),
                "sourcePath": path,
                "sourceType": "image",
                "targetDir": targetDir,
                "filename": filename
            }
            images.append(image)
    
    total_faces = 0
    for i, image in enumerate(images):
        print(f"[INFO] Processing image {i + 1}/{len(images)}")
        
        array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(array)
        
        faces = FaceDetector.detect(image["file"])
        
        for j, face in enumerate(faces):
            bbox = face['bounding_box']
            pivotX, pivotY = face['pivot']
            
            left = pivotX - bbox['width'] / 2.0 * padding
            top = pivotY - bbox['height'] / 2.0 * padding
            right = pivotX + bbox['width'] / 2.0 * padding
            bottom = pivotY + bbox['height'] / 2.0 * padding
            
            cropped = img.crop((left, top, right, bottom))
            
            os.makedirs(image['targetDir'], exist_ok=True)
            
            if image["sourceType"] == "video":
                targetFilename = f'{image["filename"]}_{i:04d}_{j}.jpg'
            else:
                targetFilename = f'{image["filename"]}_{j}.jpg'
            
            outputPath = os.path.join(image['targetDir'], targetFilename)
            
            cropped.save(outputPath)
            total_faces += 1
    
    print(f"[INFO] Found {total_faces} faces using the face detector")






def compare_faces_in_folder(folder_path):
    # Elenco dei modelli di riconoscimento supportati
    model_names = [
        "Facenet512"
    ]
    
    # Ottieni la lista dei file di immagine nella cartella
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Crea una cartella per memorizzare le immagini organizzate
    organized_folder_path = os.path.join(folder_path, "organized")
    if not os.path.exists(organized_folder_path):
        os.makedirs(organized_folder_path)

    # Funzione per trovare un nome univoco per la nuova cartella
    def get_unique_folder_path(base_path):
        index = 0
        while True:
            new_folder_path = os.path.join(base_path, f"person_{index}")
            if not os.path.exists(new_folder_path):
                return new_folder_path
            index += 1

    for img in images:
        img_path = os.path.join(folder_path, img)
        if not os.path.isfile(img_path):
            continue

        face_matched = False

        # Scorri le cartelle già create per cercare corrispondenze
        for person_folder in os.listdir(organized_folder_path):
            person_folder_path = os.path.join(organized_folder_path, person_folder)
            if not os.path.isdir(person_folder_path):
                continue

            for existing_img in os.listdir(person_folder_path):
                existing_img_path = os.path.join(person_folder_path, existing_img)

                duplicate_found = False
                for model in model_names:
                    try:
                        result = DeepFace.verify(img_path, existing_img_path, model_name=model, enforce_detection=True)
                        if result["verified"]:
                            duplicate_found = True
                            break  # Esci dal ciclo se uno dei modelli verifica la corrispondenza

                    except Exception as e:
                        print(f"Errore nel processare {img_path} e {existing_img_path} con modello {model}: {e}")
                        # Continua con il prossimo modello se si verifica un errore

                if duplicate_found:
                    # Sposta l'immagine nella cartella della persona corrispondente
                    shutil.move(img_path, person_folder_path)
                    face_matched = True
                    print(f"Immagine {img} è stata spostata nella cartella {person_folder}.")
                    break

            if face_matched:
                break

        if not face_matched:
            # Crea una nuova cartella per il nuovo volto
            new_person_folder = get_unique_folder_path(organized_folder_path)
            os.makedirs(new_person_folder)
            shutil.move(img_path, new_person_folder)
            print(f"Immagine {img} non ha corrispondenze. Creata nuova cartella {new_person_folder}.")

    print("Confronto delle immagini completato e immagini organizzate.")


def is_face_recognized(image_path):
    try:
        # Attempt to detect and analyze faces in the image
        analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=True)
        return True if analysis else False
    except:
        # If any error occurs, return False
        return False

def delete_unrecognized_faces(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            if not is_face_recognized(file_path):
                print(f"No recognizable face detected in {filename}. Deleting the file.")
                os.remove(file_path)
            else:
                print(f"Face recognized in {filename}. Keeping the file.")


def empty_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"La cartella {folder_path} non esiste.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Non sono riuscito a cancellare {file_path}. Ragione: {e}")

    print(f"La cartella {folder_path} è stata svuotata.")
    
def getFiles(path):
  files = list()
  if os.path.isdir(path):
    dirFiles = os.listdir(path)
    for file in dirFiles:
      filePath = os.path.join(path, file)
      if os.path.isdir(filePath):
        files = files + getFiles(filePath)
      else:
        kind = ft.guess(filePath)
        basename = os.path.basename(filePath)
        files.append({
          'dir': os.path.abspath(path),
          'path': filePath,
          'mime': None if kind == None else kind.mime,
          'filename': os.path.splitext(basename)[0]
        })
  else:
    kind = ft.guess(path)
    basename = os.path.basename(path)
    files.append({
      'dir': os.path.abspath(os.path.dirname(path)),
      'path': path,
      'mime': None if kind == None else kind.mime,
      'filename': os.path.splitext(basename)[0]
    })
              
  return files

def main(args):
  input = args["input"]
  output = args["output"]
  padding = float(args["padding"])

  files = getFiles(args['input'])

  inputDir = os.path.abspath(os.path.dirname(input)) if os.path.isfile(input) else os.path.abspath(input)
  outputDir = os.path.abspath(output)
  

  #empty_folder(outputDir)
  #check_faces(files, inputDir, outputDir, padding);
  #delete_unrecognized_faces(outputDir) 
  compare_faces_in_folder(outputDir) 
  #face_recognition(outputDir)
  #json_file_path = "duplicates.json"  # Sostituisci con il percorso effettivo del tuo file JSON
  #rename_files_from_duplicates(json_file_path, outputDir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-o", "--output", default="output/", help="path to output directory")
  parser.add_argument("-p", "--padding", default=1.0, help="padding ratio around the face (default: 1.0)")
  
  args = vars(parser.parse_args())
  main(args)