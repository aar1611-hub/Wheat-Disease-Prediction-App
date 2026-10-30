# Wheat Disease Prediction Using MobileNetV2-Based CNN

## 1. Project Overview

Wheat is an important food crop, and diseases can affect its growth, quality, and yield. Identifying possible disease symptoms at an early stage can help farmers and agricultural professionals decide when further inspection is needed.

This project develops an image-classification application using a Convolutional Neural Network (CNN) based on MobileNetV2. The application accepts an uploaded wheat image and predicts whether it belongs to the **Healthy Wheat** or **Sick Wheat** class.

The application is built with Python, TensorFlow/Keras, and Streamlit.

## 2. Project Objectives

- Develop a CNN-based wheat image-classification system.
- Use transfer learning with the MobileNetV2 architecture.
- Classify images into Healthy Wheat and Sick Wheat categories.
- Provide a simple interface for uploading images and viewing predictions.
- Evaluate the model using validation metrics.

## 3. Technologies Used

- **Python 3.12:** Programming language
- **TensorFlow/Keras:** Deep learning model development and inference
- **MobileNetV2:** CNN architecture used for image classification
- **Streamlit:** Web interface
- **NumPy:** Numerical operations
- **Pillow:** Image loading and handling
- **Scikit-learn:** Model evaluation utilities
- **Matplotlib:** Visualisation during model development

## 4. Project Files

- `app_mobilenet.py` — Streamlit prediction application
- `wheat_mobilenetv2.keras` — trained model
- `mobilenet_class_names.json` — class labels
- `train_mobilenet.py` — training script, if included
- `requirements.txt` — Python dependencies
- `README.md` — installation and usage instructions

## 5. System Requirements

- Windows 10 or Windows 11
- Python 3.12, 64-bit recommended
- Internet access for installing the dependencies
- Adequate free disk space for Python packages and the model
- A modern web browser

A GPU is not required for basic prediction; inference can run on a CPU, though performance will depend on the computer.

## 6. Installation Instructions

### Step 1: Install Python

Install Python 3.12 and make sure the Python launcher is available. Verify installation in PowerShell:

`py -3.12 --version`

### Step 2: Extract the Project

Extract the supplied ZIP file into a folder, then open PowerShell in that folder.

### Step 3: Create a Virtual Environment

`py -3.12 -m venv venv`

### Step 4: Activate the Environment

`.\venv\Scripts\Activate.ps1`

If PowerShell blocks activation, use Command Prompt and run:

`venv\Scripts\activate.bat`

### Step 5: Install Dependencies

`python -m pip install --upgrade pip`

`pip install -r requirements.txt`

## 7. Running the Application

Make sure the following files are in the same project folder:

- `app_mobilenet.py`
- `wheat_mobilenetv2.keras`
- `mobilenet_class_names.json`

Run the application:

`streamlit run app_mobilenet.py`

Streamlit will display a local address in the terminal, usually `http://localhost:8501`. Open that address in a browser if it does not open automatically.

## 8. How to Use the Application

1. Open the application in the browser.
2. Upload a wheat image in a supported format such as JPG, JPEG, WEBP, AVIF or PNG.
3. Use the application's prediction control.
4. View the predicted class and the class probabilities displayed by the application.
5. Try additional images to observe the model's behaviour.

The displayed probabilities are model outputs; they should not be interpreted as guaranteed real-world confidence.

## 9. Model Performance

In the reported validation experiment, the model achieved approximately **96.59% validation accuracy** on 969 validation images.

Performance on new images may vary depending on image quality, lighting, background, wheat variety, disease symptoms, and the source of the images.

## 10. Important Limitations

This application is an educational image-classification prototype. It does not provide a confirmed agricultural diagnosis. Images classified as sick or healthy should be checked by a qualified agricultural professional before making crop-treatment decisions.

## 11. Retraining the Model

Normal prediction does not require model retraining.

To retrain the model, use `train_mobilenet.py` and the required dataset, organised into the two class folders expected by the training script. Retraining may download pretrained MobileNetV2 weights and therefore requires internet access unless those weights are already available locally.

