from google.colab import drive
drive.mount('/content/drive')

import os

data_dir = '/content/drive/MyDrive/Colab Notebooks'

import pandas as pd

df = pd.read_csv(os.path.join(data_dir, 'TotalFeatures-ISCXFlowMeter.csv'))

import os
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
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Credenciales de Google definidas directamente
GOOGLE_CREDENTIALS={"type":"service_account","project_id":"plated-entry-439816-k7","private_key_id":"b1a0cc9cbd7054ff31c705668515a4a3ce8ab96c","private_key":"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCzJBoL4f2mP4oj\nHxIIInTh5yiv1Nc5KSAYcYPTt4JBbCWuelGSc7QdGYAyXFqgN88+pXMP8Q6W3veu\nzaqxeQeRKXMZQ0YUFKzTFGRay0vPR2TolwV8yyAwHqdw2QIQs/+w4VLdf6CG8bzQ\n+R/0+RNG0S1vYmspAcq2cf4v0wYdMD/uxX52oLwl4aC0cmQyoca9DhgLbkzZAfkb\nW2z3wQHuRuN6x3ysvFXjB75yOAvDo5NFQgPXGQvVFwn08retpUj39fRBc2UOFJrF\njII9zCYaYc0+0nEEBpKwS/jOY3ifZXO58pMGQ5EHdqB1hpd5CXnJDXmJnQdrdPRW\nGKAj/nLhAgMBAAECggEAH86R6li0nPNMr1o5rSSTDLYqqPeZRC6rYnOrRKjhUYTm\nduGEgMSW7xDDTI3N0INAQp8FPEggf0S1BP5C57J1174cj9h8RZ37JppJDD7q5bGH\nNu3znToCq9CRic2aGoWfKVSEjkv2IGelDxSgqjIcTFdhIhT/Ml99UuIQEAU/oWhO\nUILYEYXWrG2dQF5DLNCZ9iR0zjpu/KRXWyBmarxs4H4KtGtkcv0s3ejTAppG0L43\nAbR8A47c37Z++L7ysZdqlwzG7neMzyhUpH758hWB6kVJ/au/frYQU4yA8Ch3zchr\nxqnI3bkuicCWXALwjofCA9PIACakrfz5PGcQCdg9zQKBgQDzilRPPkd8qFKIDiCw\nwNOuMEkDALZ48uDgdGcbWRl6MujQAu6JnGntNrToDUYohx95y20KRj0+/bnnRYR3\nN2WGHppGLuZddyjlNB8w8vj+YbJAF4a6ltNSKmKKdHx6NrdoGKiZQlaybQ9jznDC\nbBsQHZkt2nldFhl6MaozGV9VHQKBgQC8TlQTkDKRO11wFwU7le8wa5WXItYHf7Bm\ndk5nO2HFQd+CjbgSsnotbLjl8zguHRrDMsV8xjIBBiwQwVoDT7rSImsTNSQhzcDR\nmDvE0zjAjmnQ4c2Fbtfq1dTUAU88tT8lhFXZgHi4Gq2imXi4wN6galf3+Y1H97L0\nvynNlgI9lQKBgQDMjSH1EOUwMZxoRB129+6TfmDEkeOxQKZaP8qeML7yYTIkDGJX\n2LUrlWhrA0MrJRrtzEvAdnBYqPls43m4PCIcfTWsvxWj6ULDCH0uOtWhq2Lw7BGw\nRKAnggwUKHSona58U0HAv/Rblrh3ZtxUoEI2zfVrivWmlro9ZNuEYcotjQKBgCpa\n3/RqicU6+iBdPTMS3XMhr8sH7eZP5UiWsbnslGg/EdwWrmGePXb8Lnaih29v4nYn\ndF5FYjfywHSgWPPHujjLvxPZ7x+fXRCH0mHKNMiy/8AZGhY6QVyz7iQli0IXbnWs\n13aNvBmE/qtFI+9CipDAerrvKcUXROxiFzAD3sslAoGAMINn2ErpoFSspTWEyqZ6\n3xovvCaJES3tEDO/f7xBf47y+o6wqL/QEaWOo8mpBQsc7kOqkgEbfwUPJHiJlpEi\nWeGdrlIQC2R/HJm8Y2pZblawdKPnZvVkOFMm41lIAdkaekfOFLypJOza6AfZIr2v\nVck9wUlB/RLLTScNJHtw3qo=\n-----END PRIVATE KEY-----\n","client_email":"emotion@plated-entry-439816-k7.iam.gserviceaccount.com","client_id":"112922743320466360316","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/emotion%40plated-entry-439816-k7.iam.gserviceaccount.com","universe_domain":"googleapis.com"}


SCOPES = ['https://www.googleapis.com/auth/drive.file']
FILE_ID = '10TT3lXJFz3zN5WNAtXQFdKki6qjE5LJ0'  # Reemplazar con el ID de tu archivo
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

if __name__ == '__main__':
    app.run(debug=True)
