# 🩺 Diabetes Prediction Project

A comprehensive machine learning project for predicting diabetes risk using health indicators from the BRFSS 2021 dataset. This project includes data analysis, visualization, preprocessing, and a web application for real-time predictions.

## 🎯 Project Overview

This project aims to predict diabetes risk using various health indicators and lifestyle factors. The analysis is performed on the **Behavioral Risk Factor Surveillance System (BRFSS) 2021** dataset, which contains comprehensive health information from CDC surveys.

### Target Variable
- **Diabetes_012**: 
  - 0 = No Diabetes
  - 1 = Pre-Diabetes  
  - 2 = Diabetes

## 📊 Dataset

The project uses the `diabetes_012_health_indicators_BRFSS2021.csv` dataset, which includes:
- **21 health indicators** including BMI, age, physical activity, smoking habits, etc.
- **Comprehensive health metrics** such as mental health, physical health, and healthcare access
- **Lifestyle factors** like alcohol consumption, exercise habits, and dietary patterns

### Key Features Analyzed
- **BMI** (Body Mass Index)
- **Age** (Age categories)
- **Physical Activity** (Exercise habits)
- **General Health** (Self-reported health status)
- **Mental Health** (Days of poor mental health)
- **Physical Health** (Days of poor physical health)
- **High Cholesterol** (Cholesterol levels)
- **Smoking Status**
- **Healthcare Access**
- **Income Level**
- **Education Level**
- **Sex** (Gender)

## ✨ Features

### 📈 Data Analysis & Visualization
- **Comprehensive EDA** with 15+ visualization plots
- **Correlation analysis** with heatmaps
- **Distribution analysis** of target variables
- **Feature relationship exploration** across different health metrics

### 🔧 Data Preprocessing
- **Duplicate detection and removal**
- **Missing value analysis and handling**
- **Feature engineering** (age grouping, categorical encoding)
- **Data standardization** using StandardScaler
- **Outlier detection and treatment** using IQR method
- **Feature selection** based on correlation thresholds

### 🤖 Machine Learning Pipeline
- **Automated preprocessing pipeline**
- **Feature selection** based on correlation with target variable
- **Data scaling and normalization**
- **Model training preparation**

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit pyngrok graphviz
```

### Clone Repository
```bash
git clone https://github.com/yourusername/diabetes-prediction-project.git
cd diabetes-prediction-project
```

## 💻 Usage

### 1. Run the Jupyter Notebook
```bash
jupyter notebook CPS_DiabetesPredictionProject.ipynb
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```

### 3. Access via ngrok (for remote access)
The notebook includes ngrok integration for public access to the Streamlit app.

## 📁 Project Structure

```
├── CPS_DiabetesPredictionProject.ipynb  # Main analysis notebook
├── app.py                               # Streamlit web application
├── diabetes_012_health_indicators_BRFSS2021.csv  # Dataset
├── processed_features.csv               # Processed feature data
├── processed_target.csv                 # Processed target data
├── pipeline.png                         # Pipeline visualization
├── README.md                           # Project documentation
└── requirements.txt                    # Dependencies
```

## 📊 Data Analysis

### Key Visualizations Include:
1. **Target Variable Distribution** - Understanding diabetes prevalence
2. **BMI vs Diabetes Status** - Box plots showing BMI relationships
3. **Age Distribution Analysis** - Violin plots across diabetes categories
4. **Lifestyle Factor Analysis** - Physical activity, smoking, alcohol consumption
5. **Health Metrics Correlation** - Mental health, physical health impacts
6. **Socioeconomic Factors** - Income, education, healthcare access effects
7. **Correlation Heatmap** - Feature relationships visualization

### Statistical Analysis:
- **Missing value detection**
- **Duplicate identification and removal**
- **Statistical summaries** for all features
- **Correlation analysis** for feature selection

## 🔄 Model Pipeline

The project implements a comprehensive preprocessing pipeline:

1. **Load Dataset** → 2. **Preprocess Data** → 3. **Feature Engineering** → 4. **Handle Outliers** → 5. **Normalize/Scale** → 6. **Feature Selection** → 7. **Save Data** → 8. **Train Model**

### Preprocessing Steps:
- **Label Encoding** for categorical variables
- **Age Grouping** (Child, Young Adult, Adult, Middle Age, Senior)
- **Missing Value Imputation** (median for numeric, mode for categorical)
- **Standardization** using StandardScaler
- **Outlier Handling** using IQR method
- **Feature Selection** based on correlation threshold (>0.1)

## 🌐 Web Application

The Streamlit web application provides:
- **Interactive input fields** for all health indicators
- **Real-time predictions** using logistic regression
- **Visual feedback** (🟢 No Diabetes / 🔴 Diabetes Risk)
- **User-friendly interface** with proper validation

### Input Features:
- Pregnancies, Glucose Level, Blood Pressure
- Skin Thickness, Insulin Level, BMI
- Diabetes Pedigree Function, Age

## 🔍 Key Findings

Through comprehensive analysis, the project reveals:
- **BMI correlation** with diabetes risk
- **Age factor importance** in diabetes prediction
- **Physical activity benefits** in diabetes prevention
- **Socioeconomic impact** on diabetes prevalence
- **Healthcare access influence** on diabetes management

## 🛠 Technologies Used

- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **Streamlit** - Web application framework
- **Jupyter Notebook** - Interactive development environment
- **ngrok** - Public URL tunneling
- **Graphviz** - Pipeline visualization
