import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import warnings
import umap
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, csv_file, img_dir):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
        self.feature_extractor = None
        
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_file)
        df = df.dropna(subset=['days_to_harvest'])
        df = df[df['days_to_harvest'] > 0]   
        
        label_encoder = LabelEncoder()
        df['class_encoded'] = label_encoder.fit_transform(df['class'])
        
        image_features = []
        class_features = []
        targets = []
        
        for idx, row in df.iterrows():
            img_path = os.path.join(self.img_dir, row['class'], row['filename'])
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((48, 48))
                img_array = np.array(img).flatten()
                
                image_features.append(img_array)
                class_features.append(row['class_encoded'])
                targets.append(row['days_to_harvest'])
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Use dummy data for missing images
                image_features.append(np.full(48*48*3, 128))
                class_features.append(row['class_encoded'])
                targets.append(row['days_to_harvest'])
        
        X_img = np.array(image_features)
        X_class = np.array(class_features).reshape(-1, 1)
        y = np.array(targets)
        
        X = np.column_stack([X_img, X_class])
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.05, random_state=42, stratify=X_class.flatten()
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25/(0.95), random_state=42,
            stratify=X_temp[:, -1]
        )

        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data loaded: Train {self.X_train.shape}, Val {self.X_val.shape}, Test {self.X_test.shape}")
        
    def evaluate_dimensionality_reduction(self):
        """Compara PCA vs UMAP y devuelve el mejor"""
        print("Evaluating dimensionality reduction methods...")
        methods = {}

        # pca
        for n_components in [50, 100]:
            if n_components < self.X_train_scaled.shape[1]:
                pca = PCA(n_components=n_components, random_state=42)
                X_train_pca = pca.fit_transform(self.X_train_scaled)
                
                # evaluacion preliminar con regresión ridge
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train_pca, self.y_train)
                
                X_val_pca = pca.transform(self.X_val_scaled)
                val_score = ridge.score(X_val_pca, self.y_val)
                
                methods[f'PCA_{n_components}'] = {
                    'method': pca,
                    'score': val_score,
                }
        
        # umap
        umap_configs = [
            {'n_components': 50, 'n_neighbors': 15, 'min_dist': 0.1},
            {'n_components': 100, 'n_neighbors': 15, 'min_dist': 0.1},
            {'n_components': 50, 'n_neighbors': 30, 'min_dist': 0.01},
            {'n_components': 100, 'n_neighbors': 30, 'min_dist': 0.01}
        ]
        
        for i, config in enumerate(umap_configs):
            if config['n_components'] < self.X_train_scaled.shape[1]:
                umap_reducer = umap.UMAP(
                    n_components=config['n_components'],
                    n_neighbors=config['n_neighbors'],
                    min_dist=config['min_dist'],
                    random_state=42,
                    n_jobs=1
                )
                
                X_train_umap = umap_reducer.fit_transform(self.X_train_scaled)
                
                # evaluacion preliminar con regresión ridge
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train_umap, self.y_train)
                
                X_val_umap = umap_reducer.transform(self.X_val_scaled)
                val_score = ridge.score(X_val_umap, self.y_val)
                
                config_name = f"UMAP_{config['n_components']}_n{config['n_neighbors']}_d{config['min_dist']}"
                methods[config_name] = {
                    'method': umap_reducer,
                    'score': val_score,
                }
        
        best_method = max(methods.items(), key=lambda x: x[1]['score'])
        method_name, method_info = best_method
        
        print(f"\nBest dimensionality reduction method: {method_name}")
        print(f"Score (R²): {method_info['score']:.4f}")
        
        self.feature_extractor = method_info['method']

        # aplica la reducción de dims
        self.X_train_reduced = self.feature_extractor.transform(self.X_train_scaled)
        self.X_val_reduced = self.feature_extractor.transform(self.X_val_scaled)
        self.X_test_reduced = self.feature_extractor.transform(self.X_test_scaled)
        
        print(f"Reduced dimensions: {self.X_train_reduced.shape[1]}")
        
        return best_method, methods
    
    def optimize_models(self):
        print("Optimizing regression models...")
        
        models = {
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            }
        }
        
        best_models = {}
        
        for name, config in models.items():
            print(f"Optimizing {name}...")
            
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=50,
                cv=5,
                scoring='r2',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(self.X_train_reduced, self.y_train)
            
            val_pred = search.best_estimator_.predict(self.X_val_reduced)
            val_r2 = r2_score(self.y_val, val_pred)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            
            best_models[name] = {
                'model': search.best_estimator_,
                'params': search.best_params_,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'val_rmse': val_rmse
            }
            
            print(f"{name} - Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        
        best_model_name = max(best_models.keys(), key=lambda k: best_models[k]['val_r2'])
        self.best_model = best_models[best_model_name]['model']
        self.best_score = best_models[best_model_name]['val_r2']
        
        print(f"\nMejor modelo: {best_model_name}")
        print(f"Con parametros: {best_models[best_model_name]['params']}")
        
        return best_models, best_model_name
    
    def evaluate_best_model(self):
        """Evaluate the best model on test set"""
        
        test_pred = self.best_model.predict(self.X_test_reduced)
        
        test_r2 = r2_score(self.y_test, test_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        
        print(f"\nTraditional ML Final Test Results:")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        return test_r2, test_mae, test_rmse, test_pred