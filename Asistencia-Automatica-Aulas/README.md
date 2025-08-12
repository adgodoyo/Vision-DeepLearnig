# Asistencia automática en aulas mediante reconocimiento facial y seguimiento visual

Este proyecto implementa un sistema de registro de asistencia basado en visión computacional. Utiliza DeepFace para el reconocimiento facial, y SORT junto con InsightFace para el seguimiento de personas en video.

---

## 1. Resumen del problema y su impacto social

El control de asistencia en instituciones educativas puede ser tedioso, impreciso y fácil de falsificar. Este sistema propone una solución automatizada que permite registrar la asistencia de manera eficiente y confiable, usando visión por computador. Su implementación tiene un impacto social importante: reduce el tiempo administrativo, mejora la precisión del registro de asistencia y puede facilitar el análisis de datos educativos, contribuyendo a una mejor toma de decisiones académicas.

---

## 2. Descripción de la arquitectura y justificación de decisiones

La solución se divide en tres fases principales:

1. **Reconocimiento facial** con DeepFace y RetinaFace para detección de rostros y extracción de embeddings.
2. **Comparación** de rostros usando distancias de similitud para identificar coincidencias con una base previa.
3. **Seguimiento visual** con SORT e InsightFace para asignar identificadores únicos a las personas detectadas en video.

Se eligieron estos modelos por su precisión, facilidad de integración en entornos de Python, y eficiencia computacional. La arquitectura modular permite adaptabilidad a otros contextos educativos o laborales.

---

## 3. Detalle de los datasets (propios y externos)

- **Imágenes de entrenamiento propias**: Carpeta `personas_salon/` con subcarpetas por estudiante (ej. `ID01/`, `ID02/`) que contienen imágenes faciales.
- **Imágenes de prueba**: Carpeta `prueba/` con imágenes para validación y pruebas.
- **Archivos de datos**:
  - `baseVision.xlsx`: Mapea IDs a nombres.
  - `embeddings_personas_salon.csv`: Embeddings faciales generados automáticamente.

No se utilizaron datasets públicos externos. Todos los datos fueron generados por los autores con consentimiento para fines académicos.

---

## 4. Métricas empleadas y discusión de resultados

- **Distancia de coseno**: Se utiliza para medir similitud entre embeddings faciales. Se considera coincidencia si la distancia es menor a 0.35.
- **Exactitud de identificación**: Se evaluó manualmente en videos de prueba, alcanzando una precisión superior al 90% bajo buenas condiciones de iluminación.
- **Tasa de falsos positivos**: Disminuye al ajustar el umbral de similitud y mejorar la calidad de las imágenes base.

Los resultados muestran que el sistema es robusto para contextos reales siempre que se controle la iluminación y resolución.

---

## 5. Lecciones aprendidas y trabajo futuro

**Lecciones aprendidas:**
- La calidad de las imágenes influye directamente en la precisión del reconocimiento.
- El seguimiento visual en video es clave para reducir ambigüedad y mantener consistencia de identidad.
- La modularidad del sistema facilita pruebas y mejoras por separado.

**Trabajo futuro:**
- Implementar reconocimiento en tiempo real desde cámaras IP.
- Integrar notificaciones automáticas por correo o sistema institucional.
- Ampliar pruebas con más sujetos y condiciones ambientales variadas.
- Desplegar el sistema en una interfaz web o móvil para usuarios finales.


---
## 6. Pipeline Completo de Reconocimiento Facial y Seguimiento
###  Requisitos
Montar Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

###  Estructura esperada de carpetas

```
personas_salon/
├── ID01/
│   ├── imagen1.jpg
│   └── imagen2.jpg
├── ID02/
...

prueba/
├── prueba1.jpg
├── prueba2.jpg

baseVision.xlsx                 # Excel con columnas ID y nombre
embeddings_personas_salon.csv  # Se genera automáticamente
```

---

###  Fase 1: Reconocimiento Facial con DeepFace

#### 1. Detección de rostros

Se detectan los rostros con RetinaFace:

```python
detections = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface")
```

#### 2. Extracción de embeddings

```python
reps = DeepFace.represent(img_path=face_path, model_name='Facenet')
```

#### 3. Guardado de resultados

```python
df_embeddings.to_csv("embeddings_personas_salon.csv", index=False)
```

---

###  Fase 2: Matching / Comparación

Se calcula la similitud entre embeddings:

```python
dist = cosine(embedding_consulta, emb_base)
```

Se considera coincidencia si la distancia es menor a 0.35.

---

###  Fase 3: Seguimiento en Video

Se utiliza InsightFace para detección facial y SORT para seguimiento en video.

#### Pasos:

1. Descargar e instalar SORT
2. Inicializar modelos
3. Procesar videos y asociar rostros con IDs conocidos
4. Generar informes en CSV

---

##  Autores

Elissa Castellanos  
Mythili Kasibhatla
