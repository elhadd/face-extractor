import cv2
import dlib
import numpy as np

def estrai_visi_unici(video_path):
    # Inizializza il rilevatore di volti di dlib (basato sul modello HOG)
    detector = dlib.get_frontal_face_detector()

    # Inizializza un set per memorizzare i volti unici trovati nel video
    volti_unici = set()

    # Apri il video utilizzando OpenCV
    cap = cv2.VideoCapture(video_path)

    # Loop finché ci sono frame nel video
    while cap.isOpened():
        # Leggi il frame successivo
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converti il frame in scala di grigi per il rilevamento dei volti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rileva i volti nel frame corrente
        faces = detector(gray)

        # Itera sui volti rilevati
        for face in faces:
            # Estrai le coordinate del rettangolo del volto
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Assicurati che le dimensioni siano positive
            if w > 0 and h > 0:
                # Ridimensiona l'immagine del volto se necessario
                face_img = gray[y:y+h, x:x+w]
                
                # Conferma che l'immagine del volto non sia vuota
                if face_img.size != 0:
                    # Converte l'immagine del volto in una stringa per memorizzare i volti unici
                    face_str = face_img.tostring()
                    
                    # Aggiungi il volto al set dei volti unici
                    volti_unici.add(face_str)

        # Mostra il frame con i rettangoli intorno ai volti (opzionale)
        # cv2.imshow('Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Rilascia la cattura del video e chiudi le finestre (se presenti)
    cap.release()
    cv2.destroyAllWindows()

    # Ritorna i volti unici trovati
    return [np.frombuffer(face_str, dtype=np.uint8).reshape(h, w) for face_str in volti_unici]

# Esempio di utilizzo:
video_path = 'test.mp4'
voli_trovati = estrai_visi_unici(video_path)

# Salvataggio dei volti unici come immagini
for i, face_img in enumerate(voli_trovati):
    cv2.imwrite(f'volto_{i+1}.jpg', face_img)

print(f"Totale volti unici estratti: {len(voli_trovati)}")