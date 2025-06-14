import streamlit as st
import face_recognition
import numpy as np
import os
import cv2
from PIL import Image

# Config Streamlit
st.set_page_config(page_title="🧠 Reconnaissance Faciale", layout="centered")
st.title("🧠 Application de Reconnaissance Faciale")
st.markdown("Cette application compare les visages détectés à ceux enregistrés dans la base `known_faces/`.")

# 📁 Chargement des visages connus
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    st.warning(f"⚠️ Le dossier '{KNOWN_FACES_DIR}/' est introuvable. Créez-le et ajoutez des images de visages connus.")
    st.stop()
else:
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Charger l'image
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Obtenir les encodages du visage
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                st.warning(f"Aucun visage détecté dans l'image {filename}")

if len(known_face_encodings) == 0:
    st.error("❌ Aucun visage valide trouvé dans le dossier 'known_faces/'. Ajoutez des images contenant des visages.")
    st.stop()

# 📤 Téléversement de l'image
uploaded_file = st.file_uploader("📤 Téléversez une image contenant un ou plusieurs visages", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image téléversée
    input_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(input_image)
    
    # Trouver tous les visages dans l'image
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    
    if not face_locations:
        st.warning("😕 Aucun visage détecté dans l'image téléversée.")
    else:
        # Convertir RGB → BGR pour OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Voir si le visage correspond à un visage connu
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Inconnu"
            
            # Trouver la meilleure correspondance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            # Dessiner le rectangle et le nom
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convertir BGR → RGB pour Streamlit
        image_rgb_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb_result, caption="📸 Résultat de reconnaissance", use_column_width=True)