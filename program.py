import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE



def load_data():
    df1 = pd.read_csv('current_campaign.csv', sep=';')
    df2 = pd.read_csv("previous_campaign_contact_counts.csv", sep=";")
    df3 = pd.read_csv("previous_campaign_contact_gap.csv", sep=";")
    df4 = pd.read_csv("previous_campaign_last_contact_results.csv", sep=";")

    merged_data = pd.merge(df1, df2, on='client_id', how='left')
    merged_data = pd.merge(merged_data, df3, on='client_id', how='left')
    merged_data = pd.merge(merged_data, df4, on='client_id', how='left')

    df = merged_data.copy()

    # Map 'no' to 0 and 'yes' to 1 in the 'y' column
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    return df


def fill_missing_values(df):
    fillna_dict = {
        'previous_campaign_contacts_count': 0,
        'age': -1,
        'contact_date_gap': 0,
        'marital': 'unknown',
        'contact_outcome': 'no contacted'
    }
    df = df.fillna(value=fillna_dict)
    df['age'] = df['age'].replace(-57, 57)
    return df

def convert_data_types(df):
    df = df.astype({"contact_date_gap": 'int', "previous_campaign_contacts_count": 'int',
                "age": 'int', "year": 'str', "month": 'str', "day": 'str', "duration": 'float'})
    return df

def combine_columns(df):
    df['year_month'] = df['year'] + '-' + df['month']
    df['month_day'] = df['month'] + '-' + df['day']
    df['contact_contact_outcome'] = df['contact'] + '-' + df['contact_outcome']
    return df

def binning(df):
    num_bins = 5
    df['age_bin'] = pd.qcut(df['age'], num_bins, labels=False, duplicates='drop')
    df['balance_bin'] = pd.qcut(df['balance'], num_bins, labels=False, duplicates='drop')
    df['duration_bin'] = pd.qcut(df['duration'], num_bins, labels=False, duplicates='drop')
    return df

def combine_bins(df):
    df['age_balance_bin'] = df['age_bin'].astype(str) + '_' + df['balance_bin'].astype(str)
    df['duration_bin_agebin_bin'] = df['age_bin'].astype(str) + '_' + df['duration_bin'].astype(str)
    df['duration_bin_campaign_bin'] = df['age_bin'].astype(str) + '_' + df['campaign'].astype(str)
    df['age_bin_job_bin'] = df['age_bin'].astype(str) + '_' + df['job'].astype(str)
    return df


def create_mean_median_columns(df):
    columns = ['age_bin', 'education', 'contact', 'contact_outcome', 'marital']
    agg_operations = ['mean', 'median']

    for col in columns:
        for operation in agg_operations:
            new_col_name = f'{operation}_{col}_duration'
            df[new_col_name] = df.groupby(col)['duration'].transform(operation)

            balance_col = df[new_col_name]  # Corrected column name here
            df[f'diff_duration_{new_col_name}'] = np.abs(df['duration'] - balance_col)

    return df



def add_ratio_columns(df):
    num_columns = ['age', 'balance', 'day', 'duration', 'campaign',
       'previous_campaign_contacts_count', 'contact_date_gap', ],

    for col1 in num_columns:
        for col2 in num_columns:
            if col1 != col2:
                ratio_col_name = f'{col1}_to_{col2}_ratio'
                df[ratio_col_name] = np.where(df[col2] != 0, df[col1] / df[col2], 0)
    return df



def marital_feature_engineering(df):
    # Additional feature engineering for 'marital' column cause it has a big influence
    marital_mean_age = df.groupby('marital')['age'].transform('mean')
    df['marital_age_difference'] = np.abs(df['age'] - marital_mean_age)
    
    marital_mean_duration = df.groupby('marital')['duration'].transform('mean')
    df['marital_duration_ratio'] = df['duration'] / marital_mean_duration
    
    marital_mean_balance = df.groupby('marital')['balance'].transform('mean')
    df['marital_balance_ratio'] = df['balance'] / marital_mean_balance
    
    return df


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def miscellaneous_processing(df):
    df['duration'] = np.where(df['duration'] == 0, 0.000000001, df['duration'])
    df['campaign'] = np.where(df['campaign'] == 0, 0.000000001, df['campaign'])
    df['duration_campaign_ratio'] = df['campaign'] / df['duration']
    return df

def preprocess_features(df):
    df = miscellaneous_processing(df)
    df = fill_missing_values(df)
    df = convert_data_types(df)

    df = add_ratio_columns(df)
    print(df[df.isna().any(axis=1)])

    df = combine_columns(df)
    df = binning(df)
    df = combine_bins(df)
    df = create_mean_median_columns(df)
    df = marital_feature_engineering(df)
    df = encode_categorical(df)
    return df


def encode_categorical(df):

    categorical_cols = df.select_dtypes(include=[object]).columns
    print(categorical_cols)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_encoded = df.copy()
    df_encoded[categorical_cols] = enc.fit_transform(df_encoded[categorical_cols])

    return df_encoded

# Train models

def train_base_models(X_train_encoded, y_train):
    catboost_base = CatBoostClassifier(eval_metric='TotalF1', verbose=False)
    lgbm_base = LGBMClassifier(max_depth=6)
    xgb_base = XGBClassifier(n_estimators=500, eval_metric='auc')

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_encoded, y_train)

    catboost_base.fit(X_res, y_res)
    lgbm_base.fit(X_res, y_res)
    xgb_base.fit(X_res, y_res)

    return catboost_base, lgbm_base, xgb_base

def train_models(X_res, y_res):
    catboost_base, lgbm_base, xgb_base = train_base_models(X_res, y_res)
    return catboost_base, lgbm_base, xgb_base

# Ensemble models
def ensemble_models(X_res, y_res):
    estimators = [
        ('catboost', CatBoostClassifier(eval_metric='TotalF1', verbose=False)),
        ('lgbm', LGBMClassifier())
    ]
    voting_clf = VotingClassifier(estimators, voting='soft')
    voting_clf.fit(X_res, y_res)
    return voting_clf

# Evaluate models
def evaluate_models(model, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    return roc_auc, f1, report


def evaluate_and_print_results(model, model_name, X_test, y_test):
    roc_auc, f1, report = evaluate_models(model, X_test, y_test)
    print(f"{model_name}:")
    print("ROC AUC:", roc_auc)
    print("F1 Score:", f1)
    print(report)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation="vertical")
    plt.tight_layout()
    plt.show()


def perform_feature_selection(model, X_train, y_train, n_features_to_select):
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    selected_feature_indices = sorted_indices[:n_features_to_select]
    selected_features = X_train.iloc[:, selected_feature_indices]
    return selected_features.columns

def main():
    df = load_data()
    y = df['y']
    X = df.drop(['y','client_id', 'contact_date'], axis=1).copy()
    X = preprocess_features(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, shuffle=True, stratify=y)
    X_res, y_res = apply_smote(X_train, y_train)


    rfe_model = CatBoostClassifier(verbose=0) 
    n_features_to_keep = 35
    selected_features = perform_feature_selection(rfe_model, X_res, y_res, n_features_to_keep)

    print("Selected Features:", selected_features)

    # Train and evaluate the model using the selected features
    catboost_rfe = CatBoostClassifier(eval_metric='TotalF1', verbose=False)
    catboost_rfe.fit(X_res[selected_features], y_res)
    evaluate_and_print_results(catboost_rfe, "CatBoost RFE", X_test[selected_features], y_test)

    # Train models

    catboost_base, lgbm_base, xgb_base = train_models(X_res, y_res)

    # Evaluate models
    evaluate_and_print_results(catboost_base, "CatBoost", X_test, y_test)
    evaluate_and_print_results(lgbm_base, "LGBM", X_test, y_test)
    evaluate_and_print_results(xgb_base, "XGBoost", X_test, y_test)

    # Ensemble models
    voting_clf = ensemble_models(X_train, y_train)

    # Evaluate ensemble model
    evaluate_and_print_results(voting_clf, "Ensemble Model", X_test, y_test)




    plot_feature_importance(catboost_base, X_train.columns)



if __name__ == "__main__":
    main()
