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
from pathlib import Path


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

def check_faces_best(files, inputDir, outputDir, padding):
    immagini = []

    # Percorsi per le due sottocartelle
    clear_output_dir = os.path.join(outputDir, 'output_clear')
    masked_output_dir = os.path.join(outputDir, 'output_masked')

    # Crea le cartelle se non esistono
    os.makedirs(clear_output_dir, exist_ok=True)
    os.makedirs(masked_output_dir, exist_ok=True)

    for file in files:
        dir, path, mime, filename = file.values()
        relative_dir = os.path.relpath(dir, inputDir)
        clear_target_dir = os.path.join(clear_output_dir, relative_dir)
        masked_target_dir = os.path.join(masked_output_dir, relative_dir)

        if mime is None:
            continue

        if mime.startswith('video'):
            print(f'[INFO] Estrazione frame dal video: {filename}')
            best_frames = get_best_frames_per_second(path)

            for frame_index, frame in enumerate(best_frames):
                image = {
                    "file": frame,  # Utilizza direttamente il frame array
                    "sourcePath": path,
                    "sourceType": "video",
                    "clearTargetDir": clear_target_dir,
                    "maskedTargetDir": masked_target_dir,
                    "filename": filename,
                    "frame_index": frame_index
                }
                immagini.append(image)

        elif mime.startswith('image'):
            image = {
                "file": cv2.imread(path),
                "sourcePath": path,
                "sourceType": "image",
                "clearTargetDir": clear_target_dir,
                "maskedTargetDir": masked_target_dir,
                "filename": filename
            }
            immagini.append(image)

    total_faces = 0
    for i, image in enumerate(immagini):
        print(f"[INFO] Elaborazione immagine {i + 1}/{len(immagini)}")

        array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)

        # Rileva volti usando face_recognition
        face_locations = face_recognition.face_locations(array)

        for j, (top, right, bottom, left) in enumerate(face_locations):
            width = right - left
            height = bottom - top
            pivotX = left + width / 2
            pivotY = top + height / 2

            # Calcola il bounding box con padding dinamico
            left = max(0, int(pivotX - width / 2.0 * (1 + padding)))
            top = max(0, int(pivotY - height / 2.0 * (1 + padding)))
            right = min(array.shape[1], int(pivotX + width / 2.0 * (1 + padding)))
            bottom = min(array.shape[0], int(pivotY + height / 2.0 * (1 + padding)))

            # Ritaglia la regione del volto
            face_image = array[top:bottom, left:right]

            # Converti array in immagine PIL per il salvataggio
            face_pil_image = Image.fromarray(face_image)

            os.makedirs(image['clearTargetDir'], exist_ok=True)
            os.makedirs(image['maskedTargetDir'], exist_ok=True)

            if image["sourceType"] == "video":
                targetFilename = f'{image["filename"]}_frame_{image["frame_index"]:04d}_face_{j}.jpg'
                maskedFilename = f'{image["filename"]}_frame_{image["frame_index"]:04d}_face_{j}_masked.jpg'
            else:
                targetFilename = f'{image["filename"]}_face_{j}.jpg'
                maskedFilename = f'{image["filename"]}_face_{j}_masked.jpg'

            clearOutputPath = os.path.join(image['clearTargetDir'], targetFilename)
            maskedOutputPath = os.path.join(image['maskedTargetDir'], maskedFilename)

            # Salva l'immagine del volto
            face_pil_image.save(clearOutputPath)

            # Creare una copia dell'immagine originale per la mascheratura
            masked_array = array.copy()

            # Maschera i volti
            for k, (mtop, mright, mbottom, mleft) in enumerate(face_locations):
                if k != j:  # Maschera solo i volti diversi da quello corrente
                    masked_array[mtop:mbottom, mleft:mright] = 0  # Imposta i pixel a nero

            # Ritaglia la regione del volto dalla copia mascherata
            masked_face_image = masked_array[top:bottom, left:right]

            # Converti la copia mascherata in immagine PIL per il salvataggio
            masked_face_pil_image = Image.fromarray(masked_face_image)

            # Salva l'immagine del volto con maschera
            masked_face_pil_image.save(maskedOutputPath)

            total_faces += 1

    print(f"[INFO] Trovati e ritagliati {total_faces} volti utilizzando il rilevatore di volti")

def check_and_delete_files(base_directory):
    masked_directory = os.path.join(base_directory, 'output_masked')
    clear_directory = os.path.join(base_directory, 'output_clear')

    for filename in os.listdir(masked_directory):
        if filename.endswith('_masked.jpg') or filename.endswith('_masked.png'):
            masked_image_path = os.path.join(masked_directory, filename)
            
            try:
                # Load the masked image
                image = face_recognition.load_image_file(masked_image_path)
                face_locations = face_recognition.face_locations(image)
                
                # Check if exactly one face is found
                if len(face_locations) == 1:
                    print(f"Image '{filename}' is valid: contains exactly one face.")
                    continue  # Proceed to next image
                else:
                    print(f"Image '{filename}' does not meet criteria: removing...")
                    
                    base_filename = os.path.splitext(filename)[0]  # Remove extension
                    original_filename = base_filename.replace('_masked', '')  # Get non-_masked version

                    # Remove masked image
                    os.remove(masked_image_path)
                    print(f"Removed '{filename}'.")

                    # Remove non-masked image if exists in clear directory
                    original_image_path_jpg = os.path.join(clear_directory, original_filename + '.jpg')
                    original_image_path_png = os.path.join(clear_directory, original_filename + '.png')

                    if os.path.isfile(original_image_path_jpg):
                        os.remove(original_image_path_jpg)
                        print(f"Removed '{original_filename}.jpg'.")

                    if os.path.isfile(original_image_path_png):
                        os.remove(original_image_path_png)
                        print(f"Removed '{original_filename}.png'.")

                    print("")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")

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
  #padding = float(args["padding"])
  padding = 2

  files = getFiles(args['input'])

  inputDir = os.path.abspath(os.path.dirname(input)) if os.path.isfile(input) else os.path.abspath(input)
  outputDir = os.path.abspath(output)
  

  empty_folder(outputDir)
  check_faces_best(files, inputDir, outputDir, padding)
  check_and_delete_files(outputDir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-o", "--output", default="output/", help="path to output directory")
  parser.add_argument("-p", "--padding", default=1.0, help="padding ratio around the face (default: 1.0)")
  
  args = vars(parser.parse_args())
  main(args)