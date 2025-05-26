# 🧠 Vision-DeepLearnig

---

## 🎯 Objetivo General

Los proyectos buscan diseñar e implementar una solución de visión computacional basada en *deep learning*. Cada trabajo debe contener un arquitectura funcional que combine al menos dos tareas diferentes de visión por computador, reentrenando al menos uno de los componentes sobre un conjunto de datos propio.

---

## 🧩 Instrucciones de Entrega

### 1️⃣ Clonar el Repositorio Asignado por el Docente

Cada grupo debe clonar el repositorio oficial habilitado para el curso:  

```bash 
git clone https://github.com/USUARIO/TALLER_FINAL_IMPACTO_SOCIAL.git
cd TALLER_FINAL
``` 

---

### 2️⃣ Crear una Nueva Rama

Cada grupo debe trabajar en una rama nombrada de la siguiente forma:  
📌 **Formato:** `grupoX_Nombre1_Nombre2`  

Ejemplo:  

```bash 
git checkout -b grupo3_CamilaLopez_SantiagoPerez
git push origin grupo3_CamilaLopez_SantiagoPerez
``` 

---

### 3️⃣ Estructura del Proyecto

Cada equipo debe subir su trabajo dentro de una carpeta claramente identificada, con la siguiente convención de nombre:  
📌 **Formato:** `NombreProblema_Nombre1_Nombre2/`

Ejemplo:

```plaintext 
📂 TALLER_FINAL_IMPACTO_SOCIAL/
│── 📁 OcupacionTransporte_CamilaLopez_SantiagoPerez/
│   │── 📁 data/                # Dataset usado, o scripts de carga desde fuente externa
│   │── 📁 src/                 # Código fuente (Scripts/Notebook y otros artefactos como el yaml)
│   │── 📜 run_pipeline.py      # Script principal de ejecución de extremo a extremo
│   │── 📜 README.md            # Reporte técnico detallado del proyecto
│   │── 📜 requirements.txt     # Archivo con las dependencias del proyecto
│── 📁 OtroGrupo/
│── 📜 README.md                # Archivo principal del repositorio (este documento)
``` 

---

## 🧪 Ejecución del Pipeline

Desde Colab o localmente (si se desea probar fuera del entorno de evaluación), el pipeline se debe correr con:

```bash 
python run_pipeline.py
``` 

Asegúrese de comentar dentro del script principal los pasos clave: carga de datos, preprocesamiento, inferencia, visualización y métricas.

---

## 📦 Instalación de Dependencias

El archivo `requirements.txt` debe incluir todas las dependencias utilizadas. Desde Colab o entorno local:

```bash 
pip install -r requirements.txt
``` 

---

## ✅ Checklist de Verificación

| Ítem | Cumplido |
|------|----------|
| Dos tareas de visión combinadas | ✅ / ❌ |
| Uso de deep learning predominante | ✅ / ❌ |
| Dataset propio usado en el entrenamiento | ✅ / ❌ |
| Script ejecutable de inicio a fin (`run_pipeline.py`) | ✅ / ❌ |
| Estructura y nombramiento correctos del repositorio | ✅ / ❌ |
| Reporte en `README.md` con las secciones solicitadas | ✅ / ❌ |
| Dependencias claras en `requirements.txt` | ✅ / ❌ |
| Código limpio y comentado | ✅ / ❌ |
| Opcional: procesamiento de video | ✅ / ❌ |

---


1. Ultralytics YOLO Docs: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
2. Roboflow Docs: [https://docs.roboflow.com/](https://docs.roboflow.com/)
3. GitHub PyGithub Docs: [https://pygithub.readthedocs.io/](https://pygithub.readthedocs.io/)
