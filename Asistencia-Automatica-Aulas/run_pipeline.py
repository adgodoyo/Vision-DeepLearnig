import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# ## Extracción masiva de embeddings faciales desde un directorio de imágenes
# 
# Este bloque recorre una carpeta de personas organizadas por ID (una carpeta por persona) y:
# 
# 1. Detecta todos los rostros en cada imagen usando `DeepFace` (backend: `retinaface`).
# 2. Extrae y guarda los vectores de embedding de cada rostro usando el modelo `Facenet`.
# 3. Registra por cada rostro:
#    - `person_id`: carpeta (identificador) de la persona.
#    - `image_file`: nombre del archivo.
#    - `face_index`: índice del rostro dentro de la imagen.
#    - `embedding`: vector de características (usualmente de 128 dimensiones).
# 
# Este proceso es fundamental para construir una **base de datos de rostros conocidos**.
# 
# 

# Ruta a la carpeta raíz
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "src")
people_folder = os.path.join(data_dir, "personas_salon")

print(people_folder)

embeddings = []
detector_backend = "retinaface"

# Recorre todas las subcarpetas (IDs de personas)
for person_id in os.listdir(people_folder):
    person_folder = os.path.join(people_folder, person_id)

    if os.path.isdir(person_folder):
        image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in image_files:
            img_path = os.path.join(person_folder, image_file)
            print(f"📷 Procesando {image_file} (ID: {person_id})")

            try:
                detections = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend, enforce_detection=False)
                img = cv2.imread(img_path)

                for idx, face in enumerate(detections):
                    region = face["facial_area"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    cropped_face = img[y:y+h, x:x+w]

                    temp_face_path = "temp_face.jpg"
                    cv2.imwrite(temp_face_path, cropped_face)

                    reps = DeepFace.represent(img_path=temp_face_path, model_name='Facenet', enforce_detection=False)
                    embedding_vector = reps[0]["embedding"]

                    embeddings.append({
                        "person_id": person_id,
                        "image_file": image_file,
                        "face_index": idx,
                        "embedding": embedding_vector
                    })

                    print(f"  ↳ Embedding extraído (rostro {idx})")

            except Exception as e:
                print(f"⚠️ Error procesando {image_file}: {e}")

# Convertir a DataFrame
df_embeddings = pd.DataFrame([
    {
        "person_id": emb["person_id"],
        "image_file": emb["image_file"],
        "face_index": emb["face_index"],
        **{f"dim_{i}": v for i, v in enumerate(emb["embedding"])}
    }
    for emb in embeddings
])

# Guardar como CSV
csv_path = os.path.join(data_dir, "embeddings_personas_salon.csv")
df_embeddings.to_csv(csv_path, index=False)
print(f"\n✅ Embeddings guardados en: {csv_path}")


df_embeddings = pd.read_csv(csv_path)

# Cargar Excel con nombres reales
df_nombres = pd.read_excel(os.path.join(data_dir, "baseVision.xlsx"), header=0)
print(df_nombres.head())
print(df_nombres.columns)
df_nombres['ID'] = df_nombres['ID'].astype(str)

df_nombres.head()

# Primero, verifica el tipo de la columna "person_id" en df_embeddings
print("Tipo de person_id:", df_embeddings["person_id"].dtype)

# Si es necesario, convierte también "person_id" a string para que coincida con "ID"
df_embeddings["person_id"] = df_embeddings["person_id"].astype(str)

# Ahora intenta el merge nuevamente
df = pd.merge(df_embeddings, df_nombres, left_on="person_id", right_on="ID", how="left")

# Verificar el resultado
print(df.head())

# Para verificar si todas las filas tienen nombres asociados
print("Cantidad de valores nulos en columna Persona:", df["Persona"].isna().sum())


# Unir ambos por person_id
df = pd.merge(df_embeddings, df_nombres, left_on="person_id", right_on="ID", how="left")
df.head()


def calcular_distancia(embedding_consulta, df_base):
    distancias = []
    for _, row in df_base.iterrows():
        emb_base = row.iloc[3:-2].values.astype(float)  # salta columnas: image_file, face_index, person_id, y nombre
        dist = cosine(embedding_consulta, emb_base)
        distancias.append((row["person_id"], row["Persona"], row["image_file"], dist))
    return sorted(distancias, key=lambda x: x[3])


import matplotlib
matplotlib.use('Agg')

import os
import shutil
import zipfile
import urllib.request

import shutil
import zipfile
import urllib.request

# Define paths in a platform-independent way
content_dir = os.path.dirname(os.path.realpath(__file__)  # Replace with your desired directory
zip_path = os.path.join(content_dir, "sort.zip")
extract_dir = os.path.join(content_dir)
sort_master_dir = os.path.join(content_dir, "sort-master")
sort_dir = os.path.join(content_dir, "sort")

# --- 1. Download the repository ---
url = "https://github.com/ElissaCQ/sort/archive/refs/heads/master.zip"
print(f"Downloading repository from {url}...")
urllib.request.urlretrieve(url, zip_path)
print("\nDownloaded repository as sort.zip.")

# --- 2. Unzip the file ---
print("\nUnzipping sort.zip...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"\nUnzipped sort.zip into {extract_dir}.")

# --- 3. CRUCIAL VERIFICATION: Check contents of the unzipped folder ---
print(f"\nDetailed contents of {sort_master_dir} (AFTER unzipping):")
for item in os.listdir(sort_master_dir):
    item_path = os.path.join(sort_master_dir, item)
    size = os.path.getsize(item_path)
    is_dir = 'dir' if os.path.isdir(item_path) else 'file'
    print(f"{is_dir:4} {size:8} {item}")

# --- 4. If structure is correct, rename and install ---
if os.path.isdir(sort_master_dir):
    print(f"\nDirectory found at {sort_master_dir}. Proceeding.")
    
    # Rename sort-master to sort for consistency
    if os.path.exists(sort_dir):
        shutil.rmtree(sort_dir)  # Remove if exists
    shutil.move(sort_master_dir, sort_dir)
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Changed directory to: {os.getcwd()}")
else:
    print(f"Error: Expected directory {sort_master_dir} not found after unzipping.")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from insightface.app import FaceAnalysis

# 1. CONFIGURACIÓN
# ----------------
print("\n1. Configurando el sistema de tracking avanzado...")

# Rutas de archivos
video_path = [
    os.path.join(data_dir, "tracking", "IMG_0140.mp4"),
    os.path.join(data_dir, "tracking", "IMG_0141.mp4")

]

output_path = [
    os.path.join(data_dir, "tracking", "processed_IMG_0140.mp4"), # Salida para IMG_0140.mp4
    os.path.join(data_dir, "tracking", "processed_IMG_0141.mp4")
]

db_path = people_folder  # Base de datos de rostros
os.makedirs("detected_faces", exist_ok=True)  # Carpeta para rostros detectados

# Verificar que exista la base de datos
if not os.path.exists(db_path):
    print("¡ERROR! La carpeta de base de datos no existe.")
    os.makedirs(db_path, exist_ok=True)
    print("Se ha creado una carpeta vacía para la demostración.")
else:
    print(f"✓ Base de datos encontrada en {db_path}")
    # Contar estudiantes registrados
    estudiantes = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
    print(f"✓ {len(estudiantes)} estudiantes registrados: {', '.join(estudiantes)}")

# Cargar nombres de estudiantes
try:
    # Intenta cargar desde Excel si está disponible
    df_nombres['ID'] = df_nombres['ID'].astype(str)
    id_to_name = dict(zip(df_nombres['ID'], df_nombres['Persona']))
    estudiantes = {id: name for id, name in id_to_name.items()}  # Actualizar diccionario de estudiantes
    print("✓ Tabla de nombres cargada correctamente")
except Exception as e:
    print(f"No se pudo cargar la tabla de nombres: {e}")
    # Crear un diccionario con nombres genéricos
    id_to_name = {
        'ID01': 'Isabela Ruiz',
        'ID02': 'Juan Jose Zuluaga',
        'ID03': 'Juan Sebastian Pacheco',
        'ID04': 'Juan Sebastian Contreras',
        'ID05': 'Sebastian Plazas',
        'ID06': 'Natalia Lizarazo',
        'ID07': 'David Espejo',
        'ID08': 'Juan Pablo Cruz',
        'ID09': 'Valeria Gomez',
        'ID10': 'Santiago Cruz',
        'ID11': 'Julian Cardenas',
        'ID12': 'Leonardo Luengas',
        'ID13': 'Danna Quito',
        'ID14': 'Luceth Argote',
        'ID15': 'Elissa Castellanos',
        'ID16': 'Juan Manuel Mompotes',
        'ID17': 'Daniel Fernandez',
        'ID18': 'Mythili Kasibhatla'
    }
    estudiantes = id_to_name.copy()  # Usar el mismo diccionario para estudiantes
    print("✓ Usando tabla de nombres predeterminada")

# Crear directorios para resultados
results_dir = os.path.join(data_dir, "tracking", "resultados")
os.makedirs(results_dir, exist_ok=True)
print(f"✓ Directorio para resultados creado en {results_dir}")

print("\n2. Inicializando modelo InsightFace...")

# Inicializar el analizador facial de InsightFace
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # 640x640 ofrece buen balance entre velocidad y precisión
print("✓ Modelo InsightFace inicializado")

from sort.sort import *
# Inicializar el tracker SORT
tracker = Sort(max_age=30,           # Cuánto tiempo mantener un track sin detecciones
              min_hits=3,            # Cuántas detecciones para iniciar un track
              iou_threshold=0.3)     # Umbral de solapamiento
print("✓ Tracker SORT inicializado")

print("\n3. Preparando base de datos de rostros conocidos...")

# Crear base de datos de rostros conocidos
known_face_db = {}
detection_count = 0

# Recorrer cada carpeta de identidad
for identity in tqdm(estudiantes, desc="Procesando identidades"):
    identity_folder = os.path.join(db_path, identity)
    image_files = [f for f in os.listdir(identity_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    known_face_db[identity] = {"embeddings": [], "name": id_to_name.get(identity, identity)}

    for img_file in image_files:
        img_path = os.path.join(identity_folder, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ⚠️ No se pudo leer: {img_path}")
                continue

            # Detectar rostros con InsightFace
            faces = face_app.get(img)

            if len(faces) == 0:
                print(f"  ⚠️ No se detectaron rostros en: {img_path}")
                continue

            # Usar el primer rostro detectado (asumiendo una foto individual)
            face_embedding = faces[0].embedding
            known_face_db[identity]["embeddings"].append(face_embedding)
            detection_count += 1

        except Exception as e:
            print(f"  ⚠️ Error procesando {img_path}: {e}")

print(f"✓ Base de datos preparada con {detection_count} rostros de {len(known_face_db)} personas")

print("\n4. Procesando videos...")

# Convertir a lista si se proporciona un solo video
if isinstance(video_path, str):
    video_paths = [video_path]
    output_paths = [output_path]
else:
    video_paths = video_path
    # Generar nombres de salida automáticos si no se proporcionan
    if isinstance(output_path, list) and len(output_path) == len(video_paths):
        output_paths = output_path
    else:
        output_paths = []
        for vid_path in video_paths:
            # Generar nombre automático basado en el video original
            base_name = os.path.basename(vid_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_name = os.path.join(data_dir, f"output_{name_without_ext}_processed.mp4")
            output_paths.append(output_name)

# Crear directorio para diagnóstico si no existe
os.makedirs(os.path.join(data_dir, "detected_faces"), exist_ok=True)

# Para almacenar los resultados de todos los videos
all_attendance_records = []

# Procesar cada video en la lista
for video_idx, (video_path, output_path) in enumerate(zip(video_paths, output_paths)):
    print(f"\n4.{video_idx+1} Procesando video: {video_path}")
    print(f"   Salida en: {output_path}")

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"¡ERROR! No se pudo abrir el video {video_path}.")
        continue

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"✓ Video cargado: {width}x{height}, {fps} FPS, {duration:.2f} segundos")

    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"✓ Video de salida configurado en {output_path}")

    # Inicializar un nuevo tracker para cada video
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Variables de tracking específicas para este video
    track_identities = {}  # Almacena información de cada track
    frames_procesados = 0
    max_frames = min(total_frames, 600)  # Limitar a 600 frames para prueba (20 segundos a 30fps)
    sim_threshold = 0.40   # Umbral de similitud para reconocimiento (ajustable)
    detection_frequency = 2  # Cada cuántos frames realizar detección

    # Para registrar asistencia en este video
    video_attendance = {}
    video_name = os.path.basename(video_path)

    # Procesamiento principal
    pbar = tqdm(total=max_frames, desc=f"Procesando video {video_idx+1}/{len(video_paths)}")

    while frames_procesados < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Crear copia para visualización
        display_frame = frame.copy()

        # PASO 1: Detectar rostros cada ciertos frames
        if frames_procesados % detection_frequency == 0:
            try:
                # Detectar rostros con InsightFace
                faces = face_app.get(frame)

                # Convertir a formato SORT
                detections = []
                face_data = {}  # Almacenar embeddings y bboxes

                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox

                    # Asegurar que las coordenadas sean válidas
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    if x2 > x1 and y2 > y1:  # Verificar bbox válido
                        detections.append([x1, y1, x2, y2, face.det_score])

                        # Guardar embedding y bbox para reconocimiento
                        bbox_key = (x1, y1, x2, y2)
                        face_data[bbox_key] = {
                            "embedding": face.embedding,
                            "bbox": (x1, y1, x2, y2)
                        }

                # Actualizar tracker con nuevas detecciones
                if detections:
                    tracked_objects = tracker.update(np.array(detections))
                else:
                    tracked_objects = tracker.update(np.empty((0, 5)))
            except Exception as e:
                print(f"Error en detección frame {frames_procesados}: {e}")
                tracked_objects = tracker.update(np.empty((0, 5)))
        else:
            # Solo actualizar posiciones (predicción)
            tracked_objects = tracker.update(np.empty((0, 5)))

        # PASO 2: Asociar detecciones con tracks y reconocer personas
        persons_in_frame = set()

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track.astype(int)
            track_id = int(track_id)

            # Inicializar si es un nuevo track
            if track_id not in track_identities:
                track_identities[track_id] = {
                    "name": "Unknown",
                    "display_name": "Desconocido",
                    "confidence": 0,
                    "frames": [],
                    "votes": {},
                    "best_similarity": 0
                }

            # Registrar este frame para el track
            track_identities[track_id]["frames"].append(frames_procesados)

            # PASO 3: Reconocer rostros si tenemos nuevos datos
            if frames_procesados % detection_frequency == 0:
                # Buscar el embedding correspondiente a este track (por IoU)
                best_iou = 0
                best_bbox_key = None

                for bbox_key in face_data:
                    # Calcular IoU entre el track y la detección
                    det_x1, det_y1, det_x2, det_y2 = bbox_key

                    # Intersección
                    inter_x1 = max(x1, det_x1)
                    inter_y1 = max(y1, det_y1)
                    inter_x2 = min(x2, det_x2)
                    inter_y2 = min(y2, det_y2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                        track_area = (x2 - x1) * (y2 - y1)
                        union_area = det_area + track_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_bbox_key = bbox_key

                # Si encontramos una coincidencia válida
                if best_bbox_key and best_iou > 0.5:
                    face_embedding = face_data[best_bbox_key]["embedding"]

                    # Reconocimiento: comparar con base de datos
                    best_similarity = -1
                    best_identity = None

                    # Comparar con cada identidad conocida
                    for identity, data in known_face_db.items():
                        # Probar con todos los embeddings de esta identidad
                        for ref_embedding in data["embeddings"]:
                            # Calcular similitud de coseno
                            similarity = np.dot(face_embedding, ref_embedding) / (
                                np.linalg.norm(face_embedding) * np.linalg.norm(ref_embedding))

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_identity = identity

                    # Si superamos el umbral de similitud, actualizar votos
                    if best_similarity > sim_threshold:
                        # Guardar mejor imagen para diagnóstico
                        if best_similarity > track_identities[track_id].get("best_similarity", 0):
                            x1_d, y1_d, x2_d, y2_d = face_data[best_bbox_key]["bbox"]
                            face_img = frame[y1_d:y2_d, x1_d:x2_d]
                            face_path = os.path.join(data_dir, "detected_faces", "track_{track_id}_vid{video_idx}_sim_{best_similarity:.2f}.jpg")
                            cv2.imwrite(face_path, face_img)
                            track_identities[track_id]["best_similarity"] = best_similarity

                        # Sistema de votación
                        votes = track_identities[track_id]["votes"]
                        if best_identity not in votes:
                            votes[best_identity] = 0

                        # Votos ponderados por similitud
                        votes[best_identity] += best_similarity

                        # Determinar identidad por mayoría de votos
                        if votes:
                            top_identity = max(votes.items(), key=lambda x: x[1])[0]
                            # Actualizar solo si hay suficientes votos acumulados o alta similitud
                            vote_weight = votes[top_identity]
                            if vote_weight >= 0.8 or best_similarity > 0.6:
                                track_identities[track_id]["name"] = top_identity
                                track_identities[track_id]["display_name"] = id_to_name.get(top_identity, top_identity)
                                track_identities[track_id]["confidence"] = best_similarity

                                # Actualizar registro de asistencia para este video
                                if top_identity not in video_attendance:
                                    video_attendance[top_identity] = {
                                        "name": id_to_name.get(top_identity, top_identity),
                                        "first_seen": frames_procesados / fps,  # Tiempo en segundos
                                        "last_seen": frames_procesados / fps,
                                        "frames_detected": 1,
                                        "confidence": best_similarity,
                                        "video": video_name
                                    }
                                else:
                                    # Actualizar registro existente
                                    video_attendance[top_identity]["last_seen"] = frames_procesados / fps
                                    video_attendance[top_identity]["frames_detected"] += 1
                                    video_attendance[top_identity]["confidence"] = max(
                                        video_attendance[top_identity]["confidence"],
                                        best_similarity
                                    )

                                # Añadir a personas detectadas en este frame
                                persons_in_frame.add(top_identity)

                                print(f"Video {video_idx+1}, Frame {frames_procesados}, Track {track_id}: "
                                      f"Identificado como {top_identity} ({id_to_name.get(top_identity, top_identity)}) "
                                      f"- Similitud: {best_similarity:.2f}")

            # PASO 4: Visualizar resultados
            identity_info = track_identities[track_id]
            name = identity_info["name"]
            display_name = identity_info.get("display_name", "Desconocido")
            conf = identity_info.get("confidence", 0)

            # Color según confianza
            if name != "Unknown":
                # Verde (alta confianza) a amarillo (baja confianza)
                green = int(255 * min(1, conf * 1.5))
                red = int(255 * (1 - min(1, conf * 1.5)))
                color = (0, green, red)
            else:
                color = (0, 0, 255)  # Rojo para desconocidos

            # Dibujar bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Preparar etiqueta
            if name != "Unknown":
                label = f"{display_name} ({conf:.2f})"
            else:
                label = "Desconocido"

            # Dibujar fondo para el texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-25), (x1+text_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1-7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Información del frame
        info_text = f"Video {video_idx+1}/{len(video_paths)} | Frame: {frames_procesados}/{max_frames} | Presentes: {len(persons_in_frame)}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Lista de personas presentes
        y_pos = 60
        if persons_in_frame:
            cv2.putText(display_frame, "Estudiantes detectados:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            for idx, person_id in enumerate(sorted(persons_in_frame)):
                person_name = id_to_name.get(person_id, person_id)
                y_pos += 25
                cv2.putText(display_frame, f"- {person_name}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Guardar frame en el video de salida
        out.write(display_frame)

        # Actualizar progreso
        frames_procesados += 1
        pbar.update(1)

    # Liberar recursos
    pbar.close()
    cap.release()
    out.release()
    print(f"✓ Procesamiento de video {video_idx+1} completado: {frames_procesados} frames procesados")

    # Añadir los registros de este video al consolidado
    for person_id, data in video_attendance.items():
        attendance_record = {
            "id": person_id,
            "name": data["name"],
            "first_seen": data["first_seen"],
            "last_seen": data["last_seen"],
            "duration": data["last_seen"] - data["first_seen"],
            "frames_detected": data["frames_detected"],
            "confidence": data["confidence"],
            "video": video_name
        }
        all_attendance_records.append(attendance_record)

    # Guardar informe de asistencia para este video
    video_df = pd.DataFrame([data for person_id, data in video_attendance.items()])
    if not video_df.empty:
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_report_path = os.path.join(data_dir, "attendance_{video_base_name}.csv")
        video_df.to_csv(video_report_path, index=False)
        print(f"✓ Informe de asistencia para video {video_idx+1} guardado en: {video_report_path}")

# Crear y guardar informe consolidado de todos los videos
if all_attendance_records:
    consolidated_df = pd.DataFrame(all_attendance_records)
    consolidated_path = os.path.join(data_dir, "consolidated_attendance.csv")
    consolidated_df.to_csv(consolidated_path, index=False)
    print(f"\n✓ Informe consolidado de asistencia guardado en: {consolidated_path}")

    # Crear resumen por persona
    summary_df = consolidated_df.groupby(['id', 'name']).agg({
        'duration': 'sum',
        'frames_detected': 'sum',
        'confidence': 'mean',
        'video': lambda x: list(set(x))
    }).reset_index()

    summary_df.rename(columns={
        'duration': 'total_duration_seconds',
        'frames_detected': 'total_frames',
        'confidence': 'avg_confidence'
    }, inplace=True)

    # Guardar resumen
    summary_path = os.path.join(data_dir, "attendance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Resumen de asistencia guardado en: {summary_path}")

    # Mostrar resumen
    print("\n--- RESUMEN DE ASISTENCIA ---")
    print(summary_df[['name', 'total_duration_seconds', 'avg_confidence']])

print("\n5. Generando registro de asistencia consolidado...")
EXCLUDE_STUDENT_IDS = ["ID15", "ID18"] # Ejemplo: ["ID03", "ID05"]
# Si no quieres excluir a nadie, déjalo como lista vacía: EXCLUDE_STUDENT_IDS = []

if EXCLUDE_STUDENT_IDS:
    print(f"INFO: Se excluirán los IDs de estudiantes: {', '.join(EXCLUDE_STUDENT_IDS)}")

# 1. Filtrar los registros de asistencia detectados
# Solo conservamos los registros de estudiantes que NO están en la lista de exclusión.
if all_attendance_records: # Asegurarse de que all_attendance_records no esté vacío
    filtered_attendance_records = [
        record for record in all_attendance_records
        if record.get('id') not in EXCLUDE_STUDENT_IDS
    ]
else:
    filtered_attendance_records = [] # Si all_attendance_records estaba vacío, filtered_attendance_records también

# 2. Filtrar la lista completa de estudiantes para el cálculo de resumen
# Solo consideramos a los estudiantes de la lista de clase que NO están en la lista de exclusión.
# (Asegúrate que 'estudiantes' esté definido y sea tu diccionario {id: nombre} completo de la clase)
if 'estudiantes' in locals() and isinstance(estudiantes, dict):
    estudiantes_considerados = {
        id_est: nombre_est for id_est, nombre_est in estudiantes.items()
        if id_est not in EXCLUDE_STUDENT_IDS
    }
else:
    print("ADVERTENCIA: La variable 'estudiantes' no está definida o no es un diccionario. El resumen de asistencia podría ser incorrecto.")
    estudiantes_considerados = {} # fallback para evitar errores


if not filtered_attendance_records:
    print("No se identificó a ninguna persona (después de aplicar exclusiones) con suficiente confianza en ningún video.")
    # Si no hay datos, inicializamos attendance_df como vacío para que el resto del código no falle
    attendance_df = pd.DataFrame()
else:
    # Crear DataFrame consolidado a partir de los registros FILTRADOS
    consolidated_df = pd.DataFrame(filtered_attendance_records)

    # Agrupar por persona para obtener estadísticas consolidadas
    attendance_summary = consolidated_df.groupby(['id', 'name']).agg({
        'duration': 'sum',
        'frames_detected': 'sum',
        'confidence': 'max',
        'video': lambda x: sorted(list(set(x))) # Usar set para videos únicos, y sorted para consistencia
    }).reset_index()

    # Calcular estadísticas adicionales
    attendance_data = []

    # --- Cálculo de total_duration (MEJORADO) ---
    total_duration = 0
    for v_path in video_paths:
        if os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0 and frame_count > 0:
                    total_duration += frame_count / fps
                cap.release()
            else:
                print(f"Advertencia: No se pudo abrir el video {v_path} para calcular la duración.")
        else:
            print(f"Advertencia: El archivo de video {v_path} no fue encontrado para calcular la duración.")

    if total_duration == 0 and video_paths:
        print("Advertencia: No se pudo calcular la duración total de los videos. El % de tiempo será 0.")
    # --- Fin del cálculo de total_duration ---

    for _, row in attendance_summary.iterrows():
        # Calcular porcentaje del tiempo total
        percent_time = (row['duration'] / total_duration) * 100 if total_duration > 0 else 0

        # Criterio de presencia: Siempre "Sí" si está en el summary (es decir, fue detectado)
        # present = True # Esta línea ya no es necesaria, asignamos "Sí" directamente

        attendance_data.append({
            "ID": row['id'],
            "Estudiante": row['name'],
            "Confianza": round(row['confidence'], 2),
            "Videos": ", ".join(row['video']),
            "Cantidad de videos": len(row['video']),
            "Frames detectados": row['frames_detected'],
            "Tiempo total (seg)": round(row['duration'], 2),
            "% del tiempo total": round(percent_time, 2),
            "Presente": "Sí" # Siempre "Sí" según la lógica previamente acordada
        })

    # Crear DataFrame y ordenar por tiempo total
    attendance_df = pd.DataFrame(attendance_data)
    if not attendance_df.empty:
        attendance_df = attendance_df.sort_values("Tiempo total (seg)", ascending=False)

# Guardar registro consolidado en CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(data_dir, "tracking", f"registro_asistencia_consolidado_{timestamp}.csv")
if not attendance_df.empty: # Solo guardar si hay datos
    attendance_df.to_csv(csv_path, index=False)
    print(f"✓ Registro de asistencia consolidado guardado en {csv_path}")
else:
    print("No se generó el archivo CSV de asistencia consolidada porque no hay datos.")

# Guardar también registros por video
for video_idx, video_path_full in enumerate(video_paths):
    video_name = os.path.basename(video_path_full)
    # Filtrar registros del video actual de la lista ya procesada (filtered_attendance_records)
    video_records = [record for record in filtered_attendance_records if record['video'] == video_name]

    if video_records:
        video_df = pd.DataFrame(video_records)
        video_base_name = os.path.splitext(os.path.basename(video_path_full))[0]
        video_csv_path = os.path.join(data_dir, "tracking", f"registro_asistencia_{video_base_name}_{timestamp}.csv")
        video_df.to_csv(video_csv_path, index=False)
        print(f"✓ Registro de asistencia para video '{video_name}' ({video_idx+1}) guardado en {video_csv_path}")
    else:
        print(f"INFO: No hay registros para el video '{video_name}' después de aplicar filtros.")


# Resumen de asistencia (ahora basado en 'estudiantes_considerados')
presentes_count = 0
if not attendance_df.empty:
    presentes_count = len(attendance_df["ID"].unique()) # IDs únicos detectados en la tabla filtrada

total_estudiantes_considerados = len(estudiantes_considerados)
ausentes_count = total_estudiantes_considerados - presentes_count

print(f"\n=== RESUMEN DE ASISTENCIA (para {total_estudiantes_considerados} estudiantes considerados) ===")
print(f"Total de estudiantes considerados para el reporte: {total_estudiantes_considerados}")
print(f"Estudiantes detectados (Presentes): {presentes_count}")
print(f"Estudiantes ausentes (de los considerados): {ausentes_count}")

# Mostrar quiénes estuvieron ausentes (de los considerados)
if ausentes_count > 0:
    detectados_ids_en_df = set(attendance_df["ID"]) if not attendance_df.empty else set()

    print("\nEstudiantes ausentes (de la lista de considerados):")
    idx_ausente = 1
    for id_est, nombre_est in estudiantes_considerados.items(): # Iterar sobre los considerados
        if id_est not in detectados_ids_en_df:
            print(f"{idx_ausente}. {nombre_est} (ID: {id_est})")
            idx_ausente += 1
elif total_estudiantes_considerados > 0:
    print("\nTodos los estudiantes considerados para el reporte fueron detectados.")
else:
    print("\nNo hay estudiantes para reportar después de aplicar exclusiones.")

print("\n6. Visualizando resultados...")


if 'attendance_df' in locals() and not attendance_df.empty:
    # --- GRÁFICO 1: Cantidad de Apariciones (Frames Detectados) por Estudiante ---
    # Este gráfico utiliza attendance_df, que ya está filtrado por las exclusiones.
    plt.figure(figsize=(12, 7))

    # Ordenar para visualización por la cantidad de frames detectados
    plot_df_frames = attendance_df.sort_values("Frames detectados", ascending=True)

    bars = plt.barh(plot_df_frames["Estudiante"], plot_df_frames["Frames detectados"],
                   color='mediumpurple')

    # Añadir valores en las barras
    for bar in bars:
        width = bar.get_width()
        text_x_pos = width + 5 if width > 10 else width + 2 # Ajustar offset para el texto
        plt.text(text_x_pos, bar.get_y() + bar.get_height()/2,
                f"{int(width)}", va='center', ha='left')

    plt.xlabel("Cantidad Total de Frames Detectados")
    plt.ylabel("Estudiante")
    plt.title("Número de Frames Detectados por Estudiante (Consolidado y Filtrado)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    max_frames = plot_df_frames["Frames detectados"].max() if not plot_df_frames.empty else 10
    plt.xlim(0, max(max_frames * 1.15, 10))
    plt.tight_layout()
    plt.show()

    # --- GRÁFICO 2: Resumen General de Asistencia (Presentes vs. Ausentes) ---
    # Este gráfico ahora debe usar 'estudiantes_considerados'
    if 'estudiantes_considerados' in locals() and isinstance(estudiantes_considerados, dict):
        total_estudiantes_para_reporte = len(estudiantes_considerados)

        if total_estudiantes_para_reporte > 0: # Solo generar gráfico si hay estudiantes considerados
            # IDs de estudiantes detectados y presentes en attendance_df (que ya está filtrado)
            ids_detectados_en_df = set(attendance_df['ID'])

            # Contar presentes y ausentes basados en la lista de 'estudiantes_considerados'
            estudiantes_presentes_count = 0
            # Contamos cuántos de los 'estudiantes_considerados' están en 'ids_detectados_en_df'
            for est_id in estudiantes_considerados.keys():
                if est_id in ids_detectados_en_df:
                    estudiantes_presentes_count += 1

            estudiantes_ausentes_count = total_estudiantes_para_reporte - estudiantes_presentes_count

            labels = ['Presentes (Detectados de la Lista Considerada)', 'Ausentes (No Detectados de la Lista Considerada)']
            sizes = [estudiantes_presentes_count, estudiantes_ausentes_count]
            colors = ['lightgreen', 'lightcoral']
            explode = (0.1, 0) if estudiantes_presentes_count > 0 and estudiantes_ausentes_count > 0 else (0, 0)

            if sum(sizes) > 0: # Asegurarse de que haya algo que graficar
                plt.figure(figsize=(8, 8))
                plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * sum(sizes) / 100, p) if p > 0 else '',
                       shadow=True, startangle=90)
                plt.axis('equal')
                plt.title(f"Resumen de Asistencia (Total Estudiantes Considerados: {total_estudiantes_para_reporte})")
                plt.show()
            else:
                print("No hay datos de presentes o ausentes para el gráfico circular (todos 0).")
        else:
            print("No hay estudiantes considerados para el reporte, no se generará el gráfico de resumen.")
    else:
        print("La variable 'estudiantes_considerados' no está definida o no es un diccionario. No se puede generar el gráfico de resumen.")

elif 'attendance_df' in locals() and attendance_df.empty:
    print("La tabla de asistencia (attendance_df) está vacía después de los filtros. No se generarán gráficos.")
    # Podrías también imprimir el estado de 'estudiantes_considerados' aquí si lo deseas
    if 'estudiantes_considerados' in locals() and isinstance(estudiantes_considerados, dict):
        total_estudiantes_para_reporte = len(estudiantes_considerados)
        print(f"INFO: Hay {total_estudiantes_para_reporte} estudiantes considerados para el reporte, pero ninguno fue detectado o todos fueron filtrados.")
        if total_estudiantes_para_reporte > 0:
             # Podríamos mostrar un gráfico de pie que indique 0% presentes, 100% ausentes
            labels = ['Presentes (Detectados de la Lista Considerada)', 'Ausentes (No Detectados de la Lista Considerada)']
            sizes = [0, total_estudiantes_para_reporte] # 0 presentes, todos ausentes
            colors = ['lightgreen', 'lightcoral']
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, colors=colors,
                   autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * sum(sizes) / 100, p) if p > 0 else '',
                   shadow=True, startangle=90)
            plt.axis('equal')
            plt.title(f"Resumen de Asistencia (Total Estudiantes Considerados: {total_estudiantes_para_reporte}) - Ninguno Detectado")
            plt.show()

    else:
        print("INFO: La variable 'estudiantes_considerados' no está definida.")


else: # attendance_df no está en locals()
    print("La variable 'attendance_df' no está definida. No se pueden generar gráficos de asistencia.")
