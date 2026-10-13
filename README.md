# 📨 Detector Inteligente de Spam (RNN/LSTM + Streamlit)

Este repositorio contiene un proyecto de extremo a extremo de **Procesamiento de Lenguaje Natural (NLP)** para resolver un problema de clasificación binaria de texto en español: determinar si un mensaje o correo electrónico es **Spam / Estafa (1)** o un correo seguro/**Legítimo (0)**.

Para optimizar los tiempos de cómputo, el flujo de trabajo se dividió en un enfoque híbrido: la experimentación, vectorización y el entrenamiento de la arquitectura **LSTM (Long Short-Term Memory)** se realizaron en la nube mediante **Google Colab (aprovechando una GPU T4)**, mientras que el modelo y su diccionario de tokens resultantes se exportaron para ser ejecutados localmente a través de una interfaz web interactiva desarrollada en **Streamlit**.

## 📌 Características Principales
* **Análisis Secuencial:** A diferencia de los filtros antiguos que solo buscan palabras sueltas, la red **LSTM** analiza el orden y contexto de las oraciones para detectar patrones de estafa estructurados.
* **100% en Español:** Tanto el set de datos simulado, las pruebas de control, como la interfaz web interactiva están completamente adaptados al idioma español.
* **Despliegue Ligero:** La aplicación web local utiliza versiones optimizadas para CPU, permitiendo predicciones en milisegundos sin requerir hardware pesado.

## 📁 Estructura del Repositorio

* `RedNeuronalRecurrente_RNN.ipynb`: Cuaderno de **Google Colab** utilizado para la tokenización de textos, configuración de la capa *Embedding*, entrenamiento de la red recurrente y exportación de pesos.
* `entrenamiento_spam.py`: Script de respaldo en Python que contiene la arquitectura LSTM completa, generación de la matriz de confusión y reportes de clasificación.
* `modelo_spam_lstm.keras`: Archivo binario con los pesos guardados de la red neuronal entrenada en la nube.
* `tokenizer_config.json`: Diccionario con el índice de palabras (tokens) exportado de Keras para asegurar que la app web traduzca el texto de la misma forma que el modelo lo aprendió.
* `app_spam.py`: Aplicación web interactiva desarrollada en **Streamlit**.
* `requirements.txt`: Lista de dependencias optimizadas para la ejecución en computadoras locales.

## 🚀 Instrucciones de Ejecución Local

### 1. Prerrequisitos
Asegúrate de tener Python instalado en tu sistema (versión 3.9 a 3.11 recomendada).

### 2. Crear y Activar el Entorno Virtual
Navega con tu terminal hasta la carpeta del proyecto y ejecuta los siguientes comandos para evitar conflictos con otras librerías de tu sistema:

**En Windows (CMD o PowerShell):**
```bash
# Crear el entorno virtual llamado 'venv'
python -m venv venv

# Activar el entorno virtual
.\venv\Scripts\activate
```

**En macOS / Linux (Terminal):**
```bash
# Crear el entorno virtual llamado 'venv'
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```
*💡 Notarás que el entorno se activó porque aparecerá la etiqueta `(venv)` al inicio de tu línea de comandos.*

### 3. Instalar las dependencias
Con el entorno virtual activo, instala los requerimientos básicos ejecutando:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lanzar la Aplicación Web
Para levantar el servidor local de Streamlit y probar el filtro antispam con tus propios textos, ejecuta:
```bash
streamlit run app_spam.py
```
Automáticamente se abrirá una pestaña en tu navegador web en la dirección `http://localhost:8501`. Escribe cualquier mensaje sospechoso y observa la predicción con su nivel de certeza en tiempo real.

---
*Nota: Cuando termines de usar o evaluar la aplicación, puedes cerrar el entorno virtual de tu terminal escribiendo simplemente el comando `deactivate`.*
