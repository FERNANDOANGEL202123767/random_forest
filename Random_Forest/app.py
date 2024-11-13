from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd

# Configuración para cargar el archivo desde Google Drive
file_id = '1zuJ0fML1HvJ6YWCFj_0_VJW0B9FfilpQ'
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
df = pd.read_csv(download_url)
print(df.head())  # Opcional: muestra las primeras filas para verificar la carga

from flask import Flask, jsonify, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import io
import base64
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
            data_scaled_head = pd.DataFrame(X_train_scaled, columns=X_train.columns).head(10).to_html()
            
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
