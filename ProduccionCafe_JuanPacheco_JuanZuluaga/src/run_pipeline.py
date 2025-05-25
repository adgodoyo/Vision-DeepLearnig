import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CoffeeProductionPipeline:
    """
    Pipeline completo de dos etapas para análisis de producción de café:
    1. YOLO para detectar clusters
    2. YOLO para detectar granos en clusters
    3. CNN para predicción de maduración
    """
    
    def __init__(self, 
                 yolo_clusters_path: str,
                 yolo_grains_path: str,
                 cnn_model_path: str,
                 confidence_threshold_clusters: float = 0.7,
                 confidence_threshold_grains: float = 0.7,
                 weight_per_grain: float = 1.6):
        """
        Inicializar el pipeline completo
        
        Args:
            yolo_clusters_path: Ruta al modelo YOLO entrenado para detectar clusters
            yolo_grains_path: Ruta al modelo YOLO entrenado para detectar granos
            cnn_model_path: Ruta al modelo CNN de regresión entrenado
            confidence_threshold_clusters: Umbral de confianza para detección de clusters
            confidence_threshold_grains: Umbral de confianza para detección de granos
            weight_per_grain: Peso promedio por grano en gramos
        """
        print("🔧 Inicializando modelos...")
        
        # Cargar modelos YOLO
        self.yolo_clusters = YOLO(yolo_clusters_path)
        self.yolo_grains = YOLO(yolo_grains_path)
        
        # Configurar CNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        
        # Configuraciones
        self.conf_clusters = confidence_threshold_clusters
        self.conf_grains = confidence_threshold_grains
        self.weight_per_grain = weight_per_grain
        
        # Mapeo de clases (ajustar según tus modelos)
        self.cluster_classes = {0: 'cluster'}  # YOLO clusters
        self.grain_classes = {0: 'verde', 1: 'rojo'}  # YOLO granos
        
        # Transformaciones para CNN
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Modelos cargados en dispositivo: {self.device}")
    
    def _load_cnn_model(self, model_path: str):
        """Cargar modelo completo previamente guardado con torch.save(model, path)"""
        try:
            from harvestDLEstimator import HarvestCNN

            model = torch.load(model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None

    
    def detect_clusters(self, image_path: str) -> List[Dict]:
        """
        Etapa 1: Detectar clusters de granos en la imagen completa
        
        Args:
            image_path: Ruta a la imagen de la mata de café
            
        Returns:
            Lista de clusters detectados con sus bounding boxes
        """
        print("🔍 Etapa 1: Detectando clusters...")
        
        results = self.yolo_clusters(image_path, conf=self.conf_clusters)
        clusters_data = []
        
        for i, result in enumerate(results):
            boxes = result.boxes
            
            if boxes is not None:
                for j, box in enumerate(boxes):
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    clusters_data.append({
                        'cluster_id': j,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': self.cluster_classes.get(cls, 'unknown')
                    })
        
        print(f"   ✅ Detectados {len(clusters_data)} clusters")
        return clusters_data
    
    def detect_grains_in_clusters(self, image_path: str, clusters_data: List[Dict]) -> List[Dict]:
        """
        Etapa 2: Detectar granos individuales dentro de cada cluster,
        organizando las salidas por imagen en subdirectorios.
        """
        print("🔍 Etapa 2: Detectando granos en clusters...")

        image = cv2.imread(image_path)
        all_grains = []
        grain_counter = 0

        # Ruta base del proyecto (sube desde src)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Obtener nombre base de la imagen
        image_name = os.path.basename(image_path)
        image_base, _ = os.path.splitext(image_name)

        # Crear carpeta específica para la imagen
        image_grains_dir = os.path.join(base_dir, 'data', 'harvest_predictions', 'cherries_predicted', f'cherries_for_{image_base}')
        os.makedirs(image_grains_dir, exist_ok=True)

        for cluster in clusters_data:
            x1, y1, x2, y2 = cluster['bbox']

            # Extraer región del cluster con margen
            margin = 10
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(image.shape[1], x2 + margin)
            y2_crop = min(image.shape[0], y2 + margin)

            cluster_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]

            if cluster_crop.size == 0:
                continue

            # Guardar crop temporal para YOLO
            temp_crop_path = os.path.join(image_grains_dir, f"temp_cluster_{cluster['cluster_id']}.jpg")
            cv2.imwrite(temp_crop_path, cluster_crop)

            try:
                # Detectar granos en el cluster
                grain_results = self.yolo_grains(temp_crop_path, conf=self.conf_grains)

                for result in grain_results:
                    boxes = result.boxes

                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf.cpu().numpy())
                            cls = int(box.cls.cpu().numpy())
                            xyxy = box.xyxy.cpu().numpy().flatten()

                            gx1, gy1, gx2, gy2 = map(int, xyxy)
                            abs_x1 = x1_crop + gx1
                            abs_y1 = y1_crop + gy1
                            abs_x2 = x1_crop + gx2
                            abs_y2 = y1_crop + gy2

                            grain_crop = image[abs_y1:abs_y2, abs_x1:abs_x2]

                            if grain_crop.size > 0:
                                grain_class = self.grain_classes.get(cls, 'unknown')
                                grain_filename = f"grain_{grain_counter}_{grain_class}.jpg"
                                grain_path = os.path.join(image_grains_dir, grain_filename)

                                cv2.imwrite(grain_path, grain_crop)

                                all_grains.append({
                                    'grain_id': grain_counter,
                                    'cluster_id': cluster['cluster_id'],
                                    'image_path': grain_path,
                                    'class': grain_class,
                                    'confidence': conf,
                                    'bbox_absolute': (abs_x1, abs_y1, abs_x2, abs_y2),
                                    'bbox_relative': (gx1, gy1, gx2, gy2),
                                    'detection_date': datetime.now().strftime('%Y-%m-%d')
                                })

                                grain_counter += 1

                # Limpiar archivo temporal
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)

            except Exception as e:
                print(f"   ⚠️ Error procesando cluster {cluster['cluster_id']}: {e}")
                continue

        print(f"   ✅ Detectados {len(all_grains)} granos individuales")
        return all_grains


    
    def predict_maturation_cnn(self, grains_data: List[Dict]) -> pd.DataFrame:
        """
        Etapa 3: Predecir tiempo de maduración usando CNN
        
        Args:
            grains_data: Lista de granos detectados
            
        Returns:
            DataFrame con predicciones de maduración
        """
        print("🧠 Etapa 3: Prediciendo maduración con CNN...")
        
        if self.cnn_model is None:
            print("❌ Modelo CNN no disponible")
            return pd.DataFrame()
        
        predictions = []
        class_mapping = {'verde': 0, 'rojo': 1}
        
        for grain in grains_data:
            try:
                # Cargar y procesar imagen del grano
                image = Image.open(grain['image_path']).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Codificar clase
                if grain['class'].lower() in class_mapping:
                    class_encoded = torch.tensor([[class_mapping[grain['class'].lower()]]], 
                                               dtype=torch.float32).to(self.device)
                    
                    # Hacer predicción
                    with torch.no_grad():
                        prediction = self.cnn_model(image_tensor, class_encoded)
                        days_to_harvest = max(0, prediction.item())
                    
                    # Calcular fecha estimada de cosecha
                    detection_date = pd.to_datetime(grain['detection_date'])
                    harvest_date = detection_date + timedelta(days=days_to_harvest)
                    
                    predictions.append({
                        'grain_id': grain['grain_id'],
                        'cluster_id': grain['cluster_id'],
                        'image_path': grain['image_path'],
                        'class': grain['class'],
                        'confidence': grain['confidence'],
                        'bbox': grain['bbox_absolute'],
                        'detection_date': detection_date,
                        'days_to_harvest': days_to_harvest,
                        'estimated_harvest_date': harvest_date,
                        'weight_grams': self.weight_per_grain
                    })
                    
            except Exception as e:
                print(f"   ⚠️ Error prediciendo grano {grain['grain_id']}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        print(f"   ✅ Predicciones completadas para {len(predictions_df)} granos")
        
        return predictions_df
    
    def create_detection_visualization(self, image_path: str, 
                                     clusters_data: List[Dict], 
                                     grains_data: List[Dict],
                                     save_dir: str = '.') -> str:
        """
        Crear visualización con detecciones marcadas (solo granos), guardando con nombre personalizado.
        """
        print("🎨 Creando visualización de detección de granos...")

        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Obtener nombre base de la imagen
        image_name = os.path.basename(image_path)
        image_base, _ = os.path.splitext(image_name)
        save_path = os.path.join(save_dir, f'detection_result_{image_base}.jpg')

        # Crear figura
        plt.figure(figsize=(16, 12))
        plt.imshow(image_rgb)

        # Dibujar granos (puntos coloreados según clase)
        colors = {'verde': 'lime', 'rojo': 'red', 'unknown': 'yellow'}

        for grain in grains_data:
            x1, y1, x2, y2 = grain['bbox_absolute']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            color = colors.get(grain['class'], 'yellow')

            # Marcar centro del grano
            plt.plot(center_x, center_y, 'o', color=color, markersize=8, 
                    markeredgecolor='white', markeredgewidth=1)

            # Bounding box (opcional)
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                            fill=False, color=color, linewidth=1, alpha=0.6))

        # Configurar plot
        plt.title(f'Detección de Granos de Café\n'
                f'Clusters: {len(clusters_data)} | Granos: {len(grains_data)}', 
                fontsize=16, weight='bold')
        plt.axis('off')

        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                    markersize=8, label='Granos Verdes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markersize=8, label='Granos Rojos')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        # Guardar imagen
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"   ✅ Visualización guardada en: {save_path}")
        return save_path
    
    def calculate_production_forecast(self, predictions_df: pd.DataFrame, 
                                    target_days: int) -> Dict:
        """Calcular pronóstico de producción para días específicos"""
        today = datetime.now()
        target_date = today + timedelta(days=target_days)
        
        # Granos que estarán listos
        ready_grains = predictions_df[
            predictions_df['days_to_harvest'] <= target_days
        ].copy()
        
        total_grains = len(ready_grains)
        total_weight_kg = (total_grains * self.weight_per_grain) / 1000
        
        # Estadísticas por clase
        class_stats = ready_grains.groupby('class').agg({
            'grain_id': 'count',
            'days_to_harvest': ['mean', 'min', 'max']
        }).round(2)
        
        return {
            'forecast_date': target_date.strftime('%Y-%m-%d'),
            'target_days': target_days,
            'ready_grains': total_grains,
            'estimated_weight_kg': round(total_weight_kg, 3),
            'class_breakdown': class_stats,
            'percentage_ready': round((total_grains / len(predictions_df)) * 100, 1) if len(predictions_df) > 0 else 0
        }
    
    def generate_production_report(self, predictions_df: pd.DataFrame, 
                                 target_days: int = 45) -> Dict:
        """
        Generar reporte completo de producción
        """
        print("📊 Generando reporte de producción...")
        
        # Cálculos principales
        forecast = self.calculate_production_forecast(predictions_df, target_days)
        
        # Distribución por clase
        class_distribution = predictions_df['class'].value_counts()
        
        # Estadísticas de maduración
        maturation_stats = {
            'avg_days_to_harvest': predictions_df['days_to_harvest'].mean(),
            'min_days_to_harvest': predictions_df['days_to_harvest'].min(),
            'max_days_to_harvest': predictions_df['days_to_harvest'].max(),
            'std_days_to_harvest': predictions_df['days_to_harvest'].std()
        }
        
        # Crear gráficos
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Reporte de Producción - Análisis {target_days} días', fontsize=16)
        
        # 1. Distribución por clase
        axes[0, 0].pie(class_distribution.values, labels=class_distribution.index, 
                      autopct='%1.1f%%', colors=['red', 'green'])
        axes[0, 0].set_title('Distribución por Clase')
        
        # 2. Histograma de días hasta cosecha
        axes[0, 1].hist(predictions_df['days_to_harvest'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].axvline(target_days, color='red', linestyle='--', label=f'Objetivo: {target_days} días')
        axes[0, 1].set_xlabel('Días hasta cosecha')
        axes[0, 1].set_ylabel('Número de granos')
        axes[0, 1].set_title('Distribución de Maduración')
        axes[0, 1].legend()
        
        # 3. Granos listos por día (acumulativo)
        days_range = range(1, min(121, int(predictions_df['days_to_harvest'].max()) + 1))
        cumulative_ready = [len(predictions_df[predictions_df['days_to_harvest'] <= d]) for d in days_range]
        
        axes[1, 0].plot(days_range, cumulative_ready, linewidth=2, color='green')
        axes[1, 0].axvline(target_days, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Días desde hoy')
        axes[1, 0].set_ylabel('Granos Acumulados')
        axes[1, 0].set_title('Granos Listos (Acumulativo)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Producción estimada por clase
        production_by_class = predictions_df.groupby('class')['weight_grams'].sum() / 1000
        axes[1, 1].bar(production_by_class.index, production_by_class.values, 
                      color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_ylabel('Peso Total (kg)')
        axes[1, 1].set_title('Producción Total por Clase')
        
        plt.tight_layout()
        # Ruta base del proyecto (sube dos niveles desde /src)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))

        # Ruta deseada: data/harvest_predictions/predicted_report
        report_dir = os.path.join(base_dir, 'data', 'harvest_predictions', 'predicted_report')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'production_analysis.png')
        
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'fecha_reporte': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_granos_detectados': len(predictions_df),
            'distribucion_clases': class_distribution.to_dict(),
            'estadisticas_maduracion': maturation_stats,
            'forecast_objetivo': forecast,
            'produccion_total_estimada_kg': round((len(predictions_df) * self.weight_per_grain) / 1000, 3)
        }
    
    def run_complete_pipeline(self, image_path: str, target_days: int = 45, 
                          output_dir: str = '.') -> Dict:
        """
        Ejecutar pipeline completo desde imagen hasta reporte final
        
        Args:
            image_path: Ruta a imagen de la mata de café
            target_days: Días para pronóstico de cosecha
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con todos los resultados
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DE ANÁLISIS DE CAFÉ")
        print("=" * 60)
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener nombre de la imagen sin extensión
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Crear subcarpeta para reportes
        report_dir = os.path.join(output_dir, 'predicted_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Etapa 1: Detectar clusters
        clusters_data = self.detect_clusters(image_path)
        
        if not clusters_data:
            print("❌ No se detectaron clusters. Terminando pipeline.")
            return {}
        
        # Etapa 2: Detectar granos en clusters
        grains_data = self.detect_grains_in_clusters(image_path, clusters_data)
        
        if not grains_data:
            print("❌ No se detectaron granos. Terminando pipeline.")
            return {}
        
        # Crear visualización de detecciones
        detection_viz_path = os.path.join(output_dir, '.')
        self.create_detection_visualization(image_path, clusters_data, grains_data, detection_viz_path)
        
        # Etapa 3: Predicciones de maduración
        predictions_df = self.predict_maturation_cnn(grains_data)
        
        if predictions_df.empty:
            print("❌ No se pudieron hacer predicciones. Terminando pipeline.")
            return {}
        
        # Guardar datos intermedios con nombre personalizado
        predictions_csv_path = os.path.join(report_dir, f'predictions_for_{image_name}.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        
        # Generar reporte final
        report = self.generate_production_report(predictions_df, target_days)
        
        # Guardar reporte JSON con nombre personalizado
        report_json_path = os.path.join(report_dir, f'production_report_for_{image_name}.json')
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Mostrar resumen final
        self._print_final_summary(clusters_data, grains_data, report, target_days)
        
        return {
            'clusters_detected': len(clusters_data),
            'grains_detected': len(grains_data),
            'predictions_df': predictions_df,
            'production_report': report,
            'visualization_path': detection_viz_path,
            'csv_path': predictions_csv_path,
            'json_path': report_json_path
        }

    
    def _print_final_summary(self, clusters_data: List[Dict], grains_data: List[Dict], 
                           report: Dict, target_days: int):
        """Imprimir resumen final del análisis"""
        print("\n" + "🎯" * 20 + " RESUMEN FINAL " + "🎯" * 20)
        print(f"📊 CLUSTERS DETECTADOS: {len(clusters_data)}")
        print(f"🌱 GRANOS DETECTADOS: {len(grains_data)}")
        
        if 'distribucion_clases' in report:
            for clase, cantidad in report['distribucion_clases'].items():
                print(f"   • {clase.capitalize()}: {cantidad} granos")
        
        print(f"\n⏰ ANÁLISIS PARA {target_days} DÍAS:")
        forecast = report.get('forecast_objetivo', {})
        print(f"   • Granos listos para cosecha: {forecast.get('ready_grains', 0)}")
        print(f"   • Producción estimada: {forecast.get('estimated_weight_kg', 0)} kg")
        print(f"   • Porcentaje del total: {forecast.get('percentage_ready', 0)}%")
        
        print(f"\n📈 PRODUCCIÓN TOTAL ESTIMADA: {report.get('produccion_total_estimada_kg', 0)} kg")
        
        print("🎯" * 56)


def main():
    # Detectar si se está ejecutando en Google Colab
    in_colab = 'google.colab' in sys.modules

    if in_colab:
        base_dir = '/content/coffee-production-dl'
    else:
        # Ir al directorio raíz del proyecto desde /src
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    yolo_clusters_path = os.path.join(base_dir, 'data', 'models', 'clusters_best.pt')
    yolo_grains_path = os.path.join(base_dir, 'data', 'models', 'cherries_best.pt')
    cnn_model_path = os.path.join(base_dir, 'data', 'models', 'days_to_harvest_best.pt')

    # Ruta de la imagen a procesar
    image_path = os.path.join(base_dir, 'data', 'harvest_predictions', 'raw_to_predict', 'IMG20250402162344.jpg')

    # Carpeta donde guardar resultados
    output_dir = os.path.join(base_dir, 'data', 'harvest_predictions')

    # Configurar pipeline
    pipeline = CoffeeProductionPipeline(
        yolo_clusters_path=yolo_clusters_path,
        yolo_grains_path=yolo_grains_path,
        cnn_model_path=cnn_model_path,
        confidence_threshold_clusters=0.5,
        confidence_threshold_grains=0.7,
        weight_per_grain=1.6
    )

    # Ejecutar pipeline completo
    results = pipeline.run_complete_pipeline(
        image_path=image_path,
        target_days=45,
        output_dir=output_dir
    )

    return results


if __name__ == "__main__":
    main()
