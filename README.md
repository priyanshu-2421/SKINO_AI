# SKINO_AI – AI-Based Skincare Recommendation System

## 📌 About the Project

**SKINO_AI** is an AI-based skincare recommendation web application developed using Python, Deep Learning, TensorFlow/Keras, MobileNetV2, and Streamlit.

The application analyzes an uploaded facial image and predicts one of four skin-type categories:

* Oily Skin
* Dry Skin
* Normal Skin
* Combination Skin

Based on the predicted skin type and user preferences, the application provides skincare guidance, product recommendations, ingredient information, and daily skincare routine suggestions.

> **Disclaimer:** This project is developed for educational purposes only. Skin-type predictions and skincare recommendations are not medical diagnoses or substitutes for professional dermatological advice.

## ✨ Features

* 📸 Facial image-based skin-type classification
* 🧴 Four skin-type categories
* 💧 Personalized skincare product recommendations
* 📝 Skin-related questionnaire
* 🧪 Skincare ingredient information guide
* ☀️ Daily skincare routine suggestions
* 🤖 Deep learning-based image classification
* 🖥️ Interactive Streamlit web interface

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow and Keras
* **Pretrained Architecture:** MobileNetV2
* **Model Technique:** Transfer Learning
* **Neural Network:** Convolutional Neural Network (CNN)
* **Web Framework:** Streamlit
* **Data Processing:** NumPy
* **Image Processing:** Pillow and OpenCV (where applicable)

## 🧠 Model Information

SKINO_AI uses the MobileNetV2 architecture with pretrained ImageNet weights and transfer learning for skin-type image classification.

| Parameter          | Details              |
| ------------------ | -------------------- |
| Model Architecture | MobileNetV2          |
| Pretrained Weights | ImageNet             |
| Input Image Size   | 224 × 224 pixels     |
| Number of Classes  | 4                    |
| Task               | Image Classification |
| Framework          | TensorFlow/Keras     |

### Output Classes

1. Combination
2. Dry
3. Normal
4. Oily

### Model Evaluation

The project's recorded evaluation results are:

| Model            | Reported Test Accuracy |
| ---------------- | ---------------------: |
| Original Model   |                 67.24% |
| Fine-Tuned Model |                 74.33% |

These results are specific to the dataset and evaluation setup used in this project.

**Current application model:** The original model is used in the application because the fine-tuned model repeatedly predicted Oily Skin during application testing.

The fine-tuned model's higher recorded test accuracy does not necessarily mean it performs better in practical use. Further evaluation is required to investigate the prediction behavior.

## 📂 Project Structure

```text
SKINO_AI/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── fine_tune_model.py
├── evaluate_finetuned_model.py
├── check_dataset.py
├── dataset_inspection.py
├── combine_datasets.py
├── prepare_dataset.py
├── augment_dataset.py
├── predict_skin.py
├── live_skin_detection.py
├── skin_care_assistant.py
├── skin_questionnaire.py
├── recommendation_engine.py
├── product_database.py
│
├── models/
│   ├── original_model.keras
│   ├── fine_tuned_model.keras
│   └── skin_type_classes.json
│
├── datasets/
├── split_data/
├── test_images/
├── training_results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The model filenames and project structure above are illustrative. Keep the actual filenames used in your project and include only files and folders that exist.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd SKINO_AI
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application should open in your default browser.

## 📊 Training and Evaluation

The repository includes scripts for dataset preparation, augmentation, model training, and evaluation.

Run these scripts only when their required datasets, dependencies, and configurations are available.

For example:

```bash
python train_model.py
```

To evaluate the original model:

```bash
python evaluate_model.py
```

To fine-tune the model:

```bash
python fine_tune_model.py
```

To evaluate the fine-tuned model:

```bash
python evaluate_finetuned_model.py
```

## ⚠️ Limitations

* Skin-type predictions depend on the quality and diversity of the training dataset.
* Lighting, camera quality, image conditions, and dataset imbalance may affect predictions.
* The model may misclassify some facial images.
* Product recommendations are informational and may not be suitable for every individual.
* The model has not been established as a clinically validated skin assessment system.

## 🔮 Future Improvements

* Improve prediction consistency across different skin types.
* Evaluate performance on a larger and more diverse dataset.
* Add class-wise precision, recall, F1-score, and a confusion matrix.
* Improve ingredient-based product filtering.
* Enhance the user interface and recommendation experience.
* Investigate the fine-tuned model's repeated Oily Skin predictions.

## 🎓 Project Purpose

SKINO_AI was developed as an educational project to explore deep learning, transfer learning, image classification, and AI-based recommendation systems.

## 📄 License

Add a suitable open-source license before distributing this project publicly. Verify the licenses of datasets and other third-party resources before redistribution.
