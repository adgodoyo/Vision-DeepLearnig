#pip install papermill

import papermill as pm
import os

# Cambia el directorio actual a src/
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

# Ejecutar Taller_aprendizaje_profundo.ipynb
pm.execute_notebook(
    'Taller_aprendizaje_profundo.ipynb',
    'Taller_aprendizaje_profundo_out.ipynb'
)

# Ejecutar segment.ipynb
pm.execute_notebook(
    'segment.ipynb',
    'segment_out.ipynb'
)