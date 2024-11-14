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

app = Flask(__name__)

# Configurar la ruta del dataset
DATA_PATH = os.path.join('data', 'archivo_optimizado.csv')

# Funciones auxiliares
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

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cada acción individual
@app.route('/action/<action>', methods=['GET'])
def action(action):
    try:
        # Leer el dataset
        df = pd.read_csv(DATA_PATH)
        
        if action == 'load_data':
            # Acción 1: Lectura y Visualización
            data_head = df.head(10).to_html()
            return jsonify({"message": "Datos cargados", "data": data_head})
        
        elif action == 'length_features':
            # Acción 2: Longitud y Características
            data_length = len(df)
            num_features = len(df.columns)
            return jsonify({"message": "Longitud y Características", "length": data_length, "features": num_features})
        
        elif action == 'split_scale':
            # Acción 3: División del Dataset y Escalado
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, 'calss')
            X_val, y_val = remove_labels(val_set, 'calss')
            X_test, y_test = remove_labels(test_set, 'calss')
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Convertir a DataFrame
            X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            data_scaled_head = X_train_scaled.head(10).to_html()
            
            return jsonify({"message": "Dataset dividido y escalado", "scaled_data": data_scaled_head})
        
        elif action == 'train_tree':
            # Acción 4: Decision Forest
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, 'calss')
            X_val, y_val = remove_labels(val_set, 'calss')
            
            clf_tree = DecisionTreeClassifier(random_state=42)
            clf_tree.fit(X_train, y_train)
            
            # Predecir con el DataSet de entrenamiento
            y_train_pred = clf_tree.predict(X_train)
            f1_train = f1_score(y_train_pred, y_train, average="weighted")
            
            # Predecir con el DataSet de Validación
            y_val_pred = clf_tree.predict(X_val)
            f1_val = f1_score(y_val_pred, y_val, average="weighted")
            
            return jsonify({"message": "Modelo entrenado", "f1_train": float(f1_train), "f1_val": float(f1_val)})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para entrenar el modelo completo y graficar
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Leer el dataset
        df = pd.read_csv(DATA_PATH)
        
        # Convertir la columna 'calss' a valores numéricos
        df['calss'], _ = pd.factorize(df['calss'])
        
        # División del dataset
        train_set, val_set, test_set = train_val_test_split(df)
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')
        X_test, y_test = remove_labels(test_set, 'calss')
        
        # Escalado
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Regresión Forestal
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Seleccionar una muestra de 5000 datos
        sample_size = 5000
        if len(X_test) > sample_size:
            X_test_sample = X_test[:sample_size]
            y_test_sample = y_test[:sample_size]
        else:
            X_test_sample = X_test
            y_test_sample = y_test
        
        # Predicciones
        y_pred = rf_model.predict(X_test_sample)
        mse = mean_squared_error(y_test_sample, y_pred)
        r2 = r2_score(y_test_sample, y_pred)
        
        # Importancia de características
        importancia = pd.DataFrame({
            'caracteristica': X_train.columns,
            'importancia': rf_model.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        # Crear gráfico
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_sample, y_pred, alpha=0.5)
        plt.plot([y_test_sample.min(), y_test_sample.max()], 
                [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        plt.tight_layout()
        
        # Guardar gráfico en buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Convertir imagen a base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        response = {
            "mse": float(mse),
            "r2": float(r2),
            "feature_importance": importancia.to_dict(orient='records'),
            "image": img_base64
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)
