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

def check_faces(files, inputDir, outputDir, padding):
    images = []
    for file in files:
        dir, path, mime, filename = file.values()

        targetDir = os.path.join(outputDir, os.path.relpath(dir, inputDir))

        if mime is None:
            continue
        
        if mime.startswith('video'):
            print('[INFO] Estrazione fotogrammi dal video...')
            
            # Apri il file video
            video = cv2.VideoCapture(path)
            
            # Lista per memorizzare le immagini estratte
            images_from_video = []

            while True:
                # Leggi un fotogramma dal video
                success, frame = video.read()
                
                # Verifica se la lettura del fotogramma è stata effettuata con successo e il fotogramma è un array numpy valido
                if success and isinstance(frame, np.ndarray):
                    # Crea un dizionario per memorizzare i dettagli del fotogramma
                    image = {
                        "file": frame,           # Dati del fotogramma
                        "sourcePath": path,      # Percorso del file video sorgente
                        "sourceType": "video",   # Tipo di sorgente
                        "targetDir": targetDir,  # Cartella di destinazione per il salvataggio dei fotogrammi
                        "filename": filename     # Nome del file per il fotogramma
                    }
                    # Aggiungi il dizionario alla lista immagini
                    images_from_video.append(image)

                else:
                    break
            
            # Aggiungi tutte le immagini estratte dal video alla lista principale delle immagini
            images.extend(images_from_video)
            video.release()
            cv2.destroyAllWindows()
        
        elif mime.startswith('image'):
            image = {
                "file": cv2.imread(path),
                "sourcePath": path,
                "sourceType": "image",
                "targetDir": targetDir,
                "filename": filename
            }
            images.append(image)

    total = 0
    for i, image in enumerate(images):
        print(f"[INFO] Elaborazione immagine {i + 1}/{len(images)}")
        
        # Rileva i volti nell'immagine utilizzando il face detector
        faces = FaceDetector.detect(image["file"])

        # Converti l'immagine da BGR a RGB per PIL
        array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(array)

        j = 1
        for face in faces:
            bbox = face['bounding_box']
            pivotX, pivotY = face['pivot']
            
            # Calcola le coordinate del bounding box ritagliato con il padding
            left = pivotX - bbox['width'] / 2.0 * padding
            top = pivotY - bbox['height'] / 2.0 * padding
            right = pivotX + bbox['width'] / 2.0 * padding
            bottom = pivotY + bbox['height'] / 2.0 * padding
            
            # Ritaglia l'immagine
            cropped = img.crop((left, top, right, bottom))

            # Crea la cartella target se non esiste
            if not os.path.exists(image['targetDir']):
                os.makedirs(image['targetDir'])
            
            # Crea il nome del file di destinazione
            if image["sourceType"] == "video":
                targetFilename = f'{image["filename"]}_{i:04d}_{j}.jpg'
            else:
                targetFilename = f'{image["filename"]}_{j}.jpg'

            outputPath = os.path.join(image['targetDir'], targetFilename)

            # Salva l'immagine ritagliata
            cropped.save(outputPath)
            total += 1
            j += 1

    print(f"[INFO] Trovati {total} volti con il face detector")

def compare_faces_in_folder(folder_path):
    # Elenco dei modelli di riconoscimento supportati
    #model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace', 'Dlib']
    model_names = ['Dlib']

    # Ottieni la lista dei file di immagine nella cartella
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)

    # Dizionari per tenere traccia delle immagini duplicate
    duplicates = {}

    # Calcola il numero totale di confronti necessari
    total_comparisons = (num_images * (num_images - 1)) // 2  
    comparison_count = 0

    for i in range(num_images):
        img1_path = os.path.join(folder_path, images[i])

        if not os.path.isfile(img1_path):
            continue

        for j in range(i + 1, num_images):
            img2_path = os.path.join(folder_path, images[j])

            if not os.path.isfile(img2_path):
                continue

            if images[i] in duplicates and images[j] in duplicates[images[i]]:
                # Le immagini sono già state confermate come duplicate, non fare ulteriori confronti
                continue

            duplicate_found = False
            for model in model_names:
                try:
                    result = DeepFace.verify(img1_path, img2_path, model_name=model, enforce_detection=False)
                    if result["verified"]:
                        # Aggiungi collegamenti bidirezionali per garantire che tutte le relazioni siano tracciate
                        if images[i] not in duplicates:
                            duplicates[images[i]] = []
                        if images[j] not in duplicates[images[i]]:
                            duplicates[images[i]].append(images[j])
                        
                        if images[j] not in duplicates:
                            duplicates[images[j]] = []
                        if images[i] not in duplicates[images[j]]:
                            duplicates[images[j]].append(images[i])
                        
                        # Stampa un messaggio quando viene trovata una corrispondenza
                        print(f"Immagine {images[i]} è duplicata con {images[j]} usando il modello {model}")
                        
                        duplicate_found = True
                        break  # Esci dal ciclo una volta trovato un duplicato con questo modello
                
                except Exception as e:
                    print(f"Errore nel processare {img1_path} e {img2_path} con modello {model}: {e}")

                # Aggiorna il contatore di confronti solo se non è stato trovato un duplicato
                if not duplicate_found:
                    comparison_count += 1
                    progress_percentage = (comparison_count / total_comparisons) * 100
                    print(f"Progresso: {progress_percentage:.2f}% completato ({comparison_count}/{total_comparisons} confronti)")

    # Salva i risultati in un file JSON
    with open("duplicates.json", "w") as outfile:
        json.dump(duplicates, outfile, indent=4)

    print("Confronto delle immagini completato. Risultati salvati in duplicates.json.")


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
  
  compare_faces_in_folder(outputDir) 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-o", "--output", default="output/", help="path to output directory")
  parser.add_argument("-p", "--padding", default=1.0, help="padding ratio around the face (default: 1.0)")
  
  args = vars(parser.parse_args())
  main(args)