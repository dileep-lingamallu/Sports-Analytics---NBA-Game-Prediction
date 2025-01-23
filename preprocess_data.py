import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data():
    # Load the cleaned merged dataset
    data = pd.read_csv('cleaned_merged_nba_data.csv')

    # Handle Missing Values
    numeric_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    data[categorical_cols] = data[categorical_cols].fillna('Unknown')

    # Encode Categorical Variables
    categorical_cols = ['HOME_TEAM', 'AWAY_TEAM']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Feature Selection
    columns_to_drop = ['GAME_ID', 'GAME_DATE_EST', 'GAME_STATUS_TEXT', 'WL']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Train/Test Split
    target_column = 'HOME_TEAM_WINS_y'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Save preprocessed data
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print("Preprocessing complete. Preprocessed datasets saved as 'X_train.csv', 'X_test.csv', 'y_train.csv', and 'y_test.csv'.")

if __name__ == "__main__":
    preprocess_data()
