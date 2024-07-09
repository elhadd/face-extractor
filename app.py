#pyinstaller --hidden-import=win32timezone your_script.py


import time

import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label  # Assicurati di importare Label
from kivy.uix.popup import Popup
from kivy.clock import Clock
import platform  # Importa il modulo platform per identificare il sistema operativo
from kivy.uix.screenmanager import ScreenManager, Screen

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from collections import defaultdict  # Importa defaultdict da collections
import filetype as ft
from deepface import DeepFace

import shutil
from pathlib import Path

import tkinter as tk
from tkinter import filedialog
import os
import winreg

class InformationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_layout()

    def create_layout(self):
        # Scritte da mostrare
        labels_text = [
            "Benvenuto a MyApp!",
            "Questa è la mia prima schermata con delle scritte.",
            "Kivy è fantastico!",
            "Puoi mostrare qualsiasi informazione qui."
        ]

        # Layout verticale per organizzare le scritte
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Creazione e aggiunta dei Label al layout
        for text in labels_text:
            label = Label(text=text, font_size='20sp', color=(0.2, 0.7, 0.9, 1))  # Esempio di personalizzazione
            layout.add_widget(label)

        # Aggiungi il layout delle scritte al layout principale della schermata
        self.add_widget(layout)
        
class MyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Trova il percorso della cartella Download
        desktop_path = self.get_desktop_path()

        # FileChooser per selezionare video o immagini
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.mpg']

        filters = []
        for ext in image_extensions:
            filters.append(f'*{ext}')
        for ext in video_extensions:
            filters.append(f'*{ext}')

        self.file_chooser = FileChooserListView(size_hint=(1, 0.8), path=desktop_path, filters=filters)
        self.layout.add_widget(self.file_chooser)

       

        # BoxLayout orizzontale per i due nuovi pulsanti
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)

        # Pulsante 1
        self.button1 = Button(text="Ricerca Volto nel Database", size_hint=(0.5, None), height=50)
        self.button1.bind(on_press=self.searchFaces)
        button_layout.add_widget(self.button1)

        # Pulsante 2
        self.scan_face_button = Button(text="Scannerizza Volti",  size_hint=(0.5, None), height=50)
        self.scan_face_button.bind(on_press=self.scanFaces)
        button_layout.add_widget(self.scan_face_button)
        
        # BoxLayout orizzontale per i due nuovi pulsanti
        button_layout2 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)
        
        # Pulsante 3
        self.button2 = Button(text="Informazioni", size_hint=(0.5, None), height=50)
        self.button2.bind(on_press=self.informationTab)
        button_layout2.add_widget(self.button2)
        
       
        # Pulsante 4
        self.button3 = Button(text="Seleziona Cartella Database", size_hint=(0.5, None), height=50)
        self.button3.bind(on_press=self.selectDatabaseFolder)
        button_layout2.add_widget(self.button3)
        self.loadDatabaseFolderPath()  # Carica la path del database all'avvio
        if self.database_folder_path:
            self.updateButtonLabel()  # Aggiorna il testo del pulsante se la path è già salvata

        self.layout.add_widget(button_layout)  # Aggiungi il button_layout al layout principale
        self.layout.add_widget(button_layout2)  # Aggiungi il button_layout al layout principale

        return self.layout

    def get_desktop_path(self):
        # Ottieni il percorso della cartella "Desktop" specifica del sistema operativo
        user_profile = os.path.expanduser('~')  # Ottieni la directory home dell'utente

        system = platform.system()  # Ottieni il nome del sistema operativo
        if system == 'Windows':
            # Windows
            desktop_path = os.path.join(user_profile, 'Desktop')
        elif system == 'Linux':
            # Linux
            desktop_path = os.path.join(user_profile, 'Desktop')
        elif system == 'Darwin':
            # macOS
            desktop_path = os.path.join(user_profile, 'Desktop')
        else:
            # Altro sistema operativo non gestito
            print(f"Sistema operativo non supportato: {system}")
            desktop_path = user_profile

        return desktop_path

    def scanFaces(self, instance):
        selected_item = self.file_chooser.selection and self.file_chooser.selection[0]
        if selected_item:
            if os.path.isfile(selected_item):
                # Verifica se il file è una immagine o un video
                if self.is_image_file(selected_item) or self.is_video_file(selected_item):
                    print(f"File selezionato: {selected_item}")
                    # Avvia il processo di elaborazione utilizzando il file selezionato
                    self.scan_face_button.disabled = True  # Disabilita il pulsante durante l'elaborazione
                    self.button1.disabled = True  # Disabilita il pulsante durante l'elaborazione
                    self.button2.disabled = True  # Disabilita il pulsante durante l'elaborazione
                    
                    input = selected_item
                    output = "output/"
                    padding = 2

                    files = self.getFiles(input)

                    inputDir = os.path.abspath(os.path.dirname(input)) if os.path.isfile(input) else os.path.abspath(input)
                    outputDir = os.path.abspath(output)

                    self.empty_folder(outputDir)
                    self.check_faces_best(files, inputDir, outputDir, padding)

                    self.check_and_delete_files(outputDir)
                    self.confronta_e_elimina_immagini(outputDir)

                    self.scan_face_button.disabled = False
                    self.button1.disabled = False  # Disabilita il pulsante durante l'elaborazione
                    self.button2.disabled = False  # Disabilita il pulsante durante l'elaborazione
                    subfolder = "output_clear"

                    # Crea il percorso completo alla sottocartella
                    subfolder_path = os.path.join(outputDir, subfolder)

                    # Apri la sottocartella usando il programma predefinito
                    os.startfile(subfolder_path)
                else:
                    print("Devi selezionare un file immagine o video.")
            else:
                print("Devi selezionare un file valido.")
        else:
            print("Seleziona un file prima di avviare l'elaborazione.")
            
#*********************SELEZIONE CARTELLA DATABASE**************************************************            
    def selectDatabaseFolder(self, instance):
        root = tk.Tk()
        root.withdraw()  # Nascondi la finestra principale di tkinter
        
        # Mostra la finestra di dialogo per la selezione della cartella
        folder_path = filedialog.askdirectory()
        
        if folder_path:
            self.database_folder_path = folder_path
            
            try:
                # Apri la chiave di registro o creala se non esiste
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\MyApp", 0, winreg.KEY_WRITE)
            except FileNotFoundError:
                key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Software\\MyApp")
            
            # Salva la path della cartella nel registro di Windows
            winreg.SetValueEx(key, "DatabaseFolder", 0, winreg.REG_SZ, folder_path)
            winreg.CloseKey(key)
            
            self.updateButtonLabel()  # Aggiorna il testo del pulsante dopo la selezione
        
    def loadDatabaseFolderPath(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\MyApp", 0, winreg.KEY_READ)
            self.database_folder_path, _ = winreg.QueryValueEx(key, "DatabaseFolder")
            winreg.CloseKey(key)
        except FileNotFoundError:
            self.database_folder_path = ""
    
    def get_database_folder_path(self):
        try:
            # Apri la chiave di registro
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\MyApp", 0, winreg.KEY_READ)
            
            # Ottieni il valore per la chiave di registro specificata
            database_folder_path, _ = winreg.QueryValueEx(key, "DatabaseFolder")
            
            # Chiudi la chiave di registro
            winreg.CloseKey(key)
            
            # Restituisci il valore ottenuto
            return database_folder_path
        except FileNotFoundError:
            # Gestisci il caso in cui la chiave o il valore di registro non esistano
            return ""
        except Exception as e:
            # Gestisci altre eccezioni potenziali e registra l'errore
            logging.error(f"Si è verificato un errore durante il caricamento del percorso della cartella del database: {e}")
            return ""
            
    def updateButtonLabel(self):
        if self.database_folder_path:
            self.button3.text = f"Cartella Database: {os.path.basename(self.database_folder_path)}"
        else:
            self.button3.text = "Seleziona Cartella Database"
#***************************************************************************************************************************************
        
    def searchFaces(self, instance):
        print("Ricerca Volto dal Database in corso...")
        self.scanFaces()
        databaseFolder = self.get_database_folder_path()
        
        

    def informationTab(self, instance):
        sm = ScreenManager()
        # Aggiungi la schermata delle informazioni al ScreenManager
        info_screen = InformationScreen(name='information')
        sm.add_widget(info_screen)

    def is_image_file(self, filename):
        # Verifica se il file è un'immagine basandosi sull'estensione del file
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def is_video_file(self, filename):
        # Verifica se il file è un video basandosi sull'estensione del file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.mpg']
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def calculate_face_visibility(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        total_face_area = 0
        for (x, y, w, h) in faces:
            total_face_area += w * h

        return total_face_area

    def calculate_image_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance

    def get_best_frames_per_second(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames_per_second = []

        frame_count = 0
        while True:
            # Imposta la posizione del video al frame del secondo successivo
            video.set(cv2.CAP_PROP_POS_MSEC, (frame_count * 1000))
            success, frame = video.read()

            if not success:
                break

            face_visibility = self.calculate_face_visibility(frame)
            image_quality = self.calculate_image_quality(frame)

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

    def check_faces_best(self, files, inputDir, outputDir, padding):
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
                best_frames = self.get_best_frames_per_second(path)

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

            detections = DeepFace.analyze(img_path=array, detector_backend='retinaface', enforce_detection=False)

            if isinstance(detections, list):
                for j, detection in enumerate(detections):
                    face_area = detection['region']
                    left, top, width, height = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                    right = left + width
                    bottom = top + height
                    pivotX = left + width / 2
                    pivotY = top + height / 2

                    left = max(0, int(pivotX - width / 2.0 * (1 + padding)))
                    top = max(0, int(pivotY - height / 2.0 * (1 + padding)))
                    right = min(array.shape[1], int(pivotX + width / 2.0 * (1 + padding)))
                    bottom = min(array.shape[0], int(pivotY + height / 2.0 * (1 + padding)))

                    face_image = array[top:bottom, left:right]
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

                    face_pil_image.save(clearOutputPath)

                    masked_array = array.copy()

                    for k, other_detection in enumerate(detections):
                        if k != j:
                            other_face_area = other_detection['region']
                            mtop, mleft = other_face_area['y'], other_face_area['x']
                            mbottom, mright = mtop + other_face_area['h'], mleft + other_face_area['w']
                            
                            # Creazione della maschera circolare
                            center = (mleft + other_face_area['w'] // 2, mtop + other_face_area['h'] // 2)
                            radius = int(min(other_face_area['w'], other_face_area['h']) / 2)
                            cv2.circle(masked_array, center, radius, (0, 0, 0), -1)

                    masked_face_image = masked_array[top:bottom, left:right]
                    masked_face_pil_image = Image.fromarray(masked_face_image)
                    masked_face_pil_image.save(maskedOutputPath)

                    total_faces += 1

        print(f"[INFO] Trovati e ritagliati {total_faces} volti utilizzando il rilevatore di volti")

    def check_and_delete_files(self, base_directory):
        masked_directory = os.path.join(base_directory, 'output_masked')
        clear_directory = os.path.join(base_directory, 'output_clear')

        for filename in os.listdir(masked_directory):
            if filename.endswith('_masked.jpg') or filename.endswith('_masked.png'):
                masked_image_path = os.path.join(masked_directory, filename)

                try:
                    # Load the masked image
                    image = cv2.imread(masked_image_path)

                    # Extract faces using DeepFace
                    detections = DeepFace.extract_faces(img_path=masked_image_path, detector_backend = 'retinaface', enforce_detection=False)

                    # Check if exactly one face is found
                    if len(detections) == 1:
                        print(f"Image '{filename}' is valid: contains exactly one face.")
                        # Optionally, you can also extract and print analysis here if needed
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

    def confronta_volto_deepface(self, file_immagine1, file_immagine2):
        try:
            # Verifica se entrambi i file immagine esistono
            if not os.path.exists(file_immagine1) or not os.path.exists(file_immagine2):
                print("Uno dei file immagine non esiste.")
                return False  # Restituisce similarità nulla se uno dei file non esiste

            # Confronta i volti e ottieni la distanza utilizzando RetinaFace come detector_backend
            #detector_backend='opencv',
            result = DeepFace.verify(img1_path=file_immagine1, img2_path=file_immagine2, detector_backend='retinaface', model_name='Facenet512')

            # Calcola la similarità come complemento della distanza
            return result['verified']

        except Exception as e:
            print(f"Errore durante il confronto dei volti con DeepFace: {str(e)}")
            return False  # In caso di errore, restituisce una similarità nulla

    def valuta_qualita_volto(self, file_immagine):
        try:
            # Carica l'immagine utilizzando OpenCV
            img = cv2.imread(file_immagine)
            
            # Calcola il punteggio di qualità del volto utilizzando OpenCV
            # In questo esempio, calcoliamo la somma della varianza dei canali BGR
            score = cv2.Laplacian(img, cv2.CV_64F).var()
            
            return score

        except Exception as e:
            print(f"Errore durante la valutazione della qualità dell'immagine: {str(e)}")
            return 0  # In caso di errore, restituisce un punteggio di qualità basso

    def confronta_e_elimina_immagini(self, cartella_immagini):
        masked_directory = os.path.join(cartella_immagini, 'output_masked')
        clear_directory = os.path.join(cartella_immagini, 'output_clear')

        try:
            # Ottenere la lista di tutti i file immagine nella cartella masked
            files = os.listdir(masked_directory)
            image_paths = [os.path.join(masked_directory, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

            i = 0
            while i < len(image_paths):
                img1_path = image_paths[i]
                
                j = i + 1
                while j < len(image_paths):
                    img2_path = image_paths[j]

                    print(f"Confrontando immagini:\n{img1_path}\n{img2_path}")

                    # Calcola la similarità tra le due immagini
                    #verifyface = 
                    #print(f"Similarità: {similarity_score}")  # Stampare la similarità qui

                    # Se la similarità è superiore alla soglia di 0.6, considera le immagini come della stessa persona
                    if self.confronta_volto_deepface(img1_path, img2_path):
                        # Valuta la qualità dell'immagine
                        score_img1 = self.valuta_qualita_volto(img1_path)
                        score_img2 = self.valuta_qualita_volto(img2_path)

                        # Mantieni l'immagine con la qualità più alta
                        if score_img1 > score_img2:
                            # Rimuovi img2 dalla cartella e dalla lista
                            if os.path.exists(img2_path):
                                os.remove(img2_path)
                                print(f"Immagine eliminata: {img2_path}")
                           
                            
                            base_filename = os.path.splitext(img2_path)[0]  # Remove extension
                            #print(f"Immagine eliminata1: {base_filename}")
                            original_filename = base_filename.replace('output_masked', 'output_clear')  # Get non-_masked version
                            original_filename = original_filename.replace('_masked', '')  # Get non-_masked version
                            # Remove non-masked image if exists in clear directory
                            #print(f"Immagine eliminata2: {original_filename}")
                            original_image_path_jpg = os.path.join(original_filename + '.jpg')
                            #print(f"Immagine eliminata3: {original_image_path_jpg}")
                            
                            if os.path.exists(original_image_path_jpg):
                                os.remove(original_image_path_jpg)
                                print(f"Immagine eliminata: {original_image_path_jpg}")
                              
                                
                        else:
                            # Rimuovi img1 dalla cartella e dalla lista
                            if os.path.exists(img1_path):
                                os.remove(img1_path)
                                print(f"Immagine eliminata: {img1_path}")
                           
                            
                            base_filename = os.path.splitext(img1_path)[0]  # Remove extension
                            original_filename = base_filename.replace('output_masked', 'output_clear')  # Get non-_masked version
                            original_filename = original_filename.replace('_masked', '')  # Get non-_masked version
                            # Remove non-masked image if exists in clear directory
                            original_image_path_jpg = os.path.join(original_filename + '.jpg')
                            
                            if os.path.exists(original_image_path_jpg):
                                os.remove(original_image_path_jpg)
                                print(f"Immagine eliminata: {original_image_path_jpg}")
                                
                    else:
                        j += 1
                
                i += 1

            print("Eliminazione delle immagini duplicate completata.")

        except Exception as e:
            print(f"Errore durante il confronto e l'eliminazione delle immagini: {str(e)}")


    def empty_folder(self, folder_path):
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

    def getFiles(self, path):
        files = list()
        if os.path.isdir(path):
            dirFiles = os.listdir(path)
            for file in dirFiles:
                filePath = os.path.join(path, file)
                if os.path.isdir(filePath):
                    files = files + self.getFiles(filePath)
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


if __name__ == '__main__':
    MyApp().run()
