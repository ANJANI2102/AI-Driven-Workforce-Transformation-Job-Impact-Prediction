import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


FEATURES = [
    'Age',
    'Gender',
    'Education_Level',
    'Industry',
    'Job_Role',
    'Years_Experience',
    'AI_Adoption_Level',
    'Automation_Risk',
    'Upskilling_Required',
    'Salary_Before_AI',
    'Salary_After_AI',
    'Work_Hours_Per_Week',
    'Remote_Work',
    'Job_Satisfaction',
    'Productivity_Change_%'
]

CATEGORICAL_COLS = [
    'Gender',
    'Education_Level',
    'Industry',
    'Job_Role',
    'AI_Adoption_Level',
    'Automation_Risk',
    'Upskilling_Required',
    'Remote_Work'
]


def load_data(file_path):

    dataset = pd.read_csv(file_path)

    X = dataset[FEATURES]
    y = dataset['Job_Status']

    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                CATEGORICAL_COLS
            )
        ],
        remainder=StandardScaler()
    )

    return preprocessor
