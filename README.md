# Sports-Analytics---NBA-Game-Prediction

This project is focused on predicting the outcome of NBA games, specifically whether the home team wins or loses, using historical game data and advanced machine learning techniques. The model aims to provide actionable insights into factors influencing game outcomes and achieve high predictive accuracy.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Introduction

Basketball games, especially NBA games, are highly competitive and influenced by various factors such as team performance, player statistics, and game conditions. This project leverages data preprocessing, feature engineering, and machine learning to predict game outcomes and uncover the most important factors contributing to home team wins.

---

## Dataset Description

### 1. NBA API Data
- Detailed statistics for games, including scores, rebounds, assists, and team matchups.
- Derived fields:
  - `HOME_TEAM`: Home team name.
  - `AWAY_TEAM`: Away team name.
  - `HOME_TEAM_WINS`: Binary field indicating whether the home team won (1) or lost (0).

### 2. Kaggle NBA Dataset
- Additional data:
  - `GAME_DATE_EST`: Game date.
  - `PTS_home`: Points scored by the home team.
  - `PTS_away`: Points scored by the away team.
  - `REB_home`, `REB_away`: Rebounds for home and away teams.

### Merged Dataset
- Combined insights from both datasets for enhanced predictions.
- Total rows: 2778.
- Total columns: 35.

---

## Project Workflow

### Step 1: Data Preprocessing
- Merged NBA API and Kaggle datasets on `GAME_ID`.
- Handled missing values:
  - Filled numeric columns with `0`.
  - Replaced missing categorical values with `"Unknown"`.
- Encoded categorical features using one-hot encoding.
- Balanced the dataset using SMOTE to address class imbalance.

### Step 2: Model Training
- Trained two machine learning models:
  1. **Random Forest Classifier**
  2. **XGBoost Classifier**
- Evaluated models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### Step 3: Model Evaluation
- **Random Forest**:
  - Accuracy: 96%.
  - Strong performance for both majority and minority classes.
- **XGBoost**:
  - Accuracy: 100%.
  - Near-perfect predictions with no false negatives or false positives.

### Step 4: Insights and Visualizations
- **Feature Importance**:
  - Identified key features like `PTS_home`, `PTS_away`, `REB_home`, and `AST_home`.
- **Confusion Matrix**:
  - Demonstrated model's accuracy in distinguishing between `HOME_TEAM_WINS = 0` and `HOME_TEAM_WINS = 1`.
- **ROC Curve**:
  - Achieved an AUC score of 1.00, highlighting perfect model performance.

---

## Key Results

- **Best Model**: XGBoost Classifier.
- **Performance Metrics**:
  - Accuracy: 100%.
  - Precision/Recall for Class 0: 1.00.
  - Precision/Recall for Class 1: 1.00.
- **Feature Importance**:
  - Most influential features include team performance metrics like points scored, rebounds, and assists.

---

## Technologies Used

- **Languages**: Python
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`, `xgboost`, `imblearn`
  - Visualization: `matplotlib`, `seaborn`

---

## How to Run

### Prerequisites
1. Python 3.9 or higher.
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nba-game-prediction.git
   cd nba-game-prediction
   ```
2. Run the preprocessing script:
   ```bash
   python preprocess_data.py
   ```
3. Train and evaluate the model:
   ```bash
   python train_model.py
   ```
4. Generate visualizations:
   ```bash
   python visualize_results.py
   ```

---

## Visualizations

1. **Feature Importance**:
   - Highlights the top 10 features contributing to predictions.

2. **Confusion Matrix**:
   - Visualizes the model's ability to distinguish between home team wins and losses.

3. **ROC Curve**:
   - Demonstrates the model's classification performance with an AUC score of 1.00.

---

## Future Improvements

1. **Expand Dataset**:
   - Include player-level statistics and advanced metrics for deeper insights.
2. **Advanced Models**:
   - Experiment with neural networks for even higher accuracy.
3. **Interactive Dashboard**:
   - Develop a dashboard using Power BI or Tableau for real-time predictions and visualizations.
4. **Deployment**:
   - Deploy the model using Flask or FastAPI for real-time predictions.

---

