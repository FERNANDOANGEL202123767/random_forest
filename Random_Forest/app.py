from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt
import io
import base64
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
from dotenv import load_dotenv
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cargar las variables del archivo .env
load_dotenv()

# Obtener las credenciales de Google desde el archivo .env
GOOGLE_CREDENTIALS = json.loads(os.getenv('GOOGLE_CREDENTIALS'))
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FILE_ID = '12CKEe1qAXXcUgQj0Xj2IRZbvbysE00aU'  # Reemplazar con el ID del archivo en Drive

def obtener_servicio_drive():
    """Inicializar el servicio de Google Drive"""
    try:
        creds = service_account.Credentials.from_service_account_info(
            GOOGLE_CREDENTIALS, 
            scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Error al inicializar el servicio de Drive: {str(e)}")
        raise

def get_csv_from_drive():
    """Descargar y leer el archivo CSV desde Google Drive"""
    try:
        service = obtener_servicio_drive()
        request = service.files().get_media(fileId=FILE_ID)
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        
        while not done:
            status, done = downloader.next_chunk()
            
        file.seek(0)
        df = pd.read_csv(file)
        return df
        
    except Exception as e:
        logger.error(f"Error al obtener el archivo CSV de Drive: {str(e)}")
        raise

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/action/<action>', methods=['GET'])
def action(action):
    try:
        # Leer el dataset desde Drive
        df = get_csv_from_drive()
        
        if action == 'load_data':
            data_head = df.head(10).to_html()
            return jsonify({"message": "Datos cargados desde Drive", "data": data_head})
        
        elif action == 'length_features':
            data_length = len(df)
            num_features = len(df.columns)
            return jsonify({"message": "Longitud y Características", "length": data_length, "features": num_features})
        
        elif action == 'split_scale':
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, 'calss')
            X_val, y_val = remove_labels(val_set, 'calss')
            X_test, y_test = remove_labels(test_set, 'calss')
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            data_scaled_head = X_train_scaled.head(10).to_html()
            
            return jsonify({"message": "Dataset dividido y escalado", "scaled_data": data_scaled_head})
        
        elif action == 'train_tree':
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, 'calss')
            X_val, y_val = remove_labels(val_set, 'calss')
            
            clf_tree = DecisionTreeClassifier(random_state=42)
            clf_tree.fit(X_train, y_train)
            
            y_train_pred = clf_tree.predict(X_train)
            f1_train = f1_score(y_train_pred, y_train, average="weighted")
            
            y_val_pred = clf_tree.predict(X_val)
            f1_val = f1_score(y_val_pred, y_val, average="weighted")
            
            return jsonify({"message": "Modelo entrenado", "f1_train": float(f1_train), "f1_val": float(f1_val)})
            
    except Exception as e:
        logger.error(f"Error en la acción {action}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Leer el dataset desde Drive
        df = get_csv_from_drive()
        
        # Convertir la columna 'calss' a valores numéricos
        df['calss'], _ = pd.factorize(df['calss'])
        
        train_set, val_set, test_set = train_val_test_split(df)
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')
        X_test, y_test = remove_labels(test_set, 'calss')
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestRegressor(
            n_estimators=5,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        sample_size = 5000
        if len(X_test) > sample_size:
            X_test_sample = X_test[:sample_size]
            y_test_sample = y_test[:sample_size]
        else:
            X_test_sample = X_test
        
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        importancia = pd.DataFrame({
            'caracteristica': X_train.columns,
            'importancia': rf_model.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        response = {
            "mse": float(mse),
            "r2": float(r2),
            "feature_importance": importancia.to_dict(orient='records'),
            "image": img_base64
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento del modelo: {str(e)}")
        return jsonify({"error": str(e)}), 500
