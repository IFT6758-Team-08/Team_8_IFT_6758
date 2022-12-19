# Import libraries for data processing and modeling
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from comet_ml import Experiment
import joblib

# Load the data
def load_data(path):
    df = pd.read_csv(path) # Path to the result of Feature Engineering 1 data
    return df

def analyze_data(df):
    # Analyzing correlation between all the features using heatmap
    corr = df.corr()
    plt.figure(figsize=(5,5))
    sns.heatmap(corr, annot=True, fmt='.1g')
    plt.show()  

def XGBoost1(x_train, x_val,y_train, y_val):
    # XGBoost on distance and angle
    # X = df[['shot_distance','shot_angle']].astype(np.float32)
    # y = df['goal']

    # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    xgb1 = xgboost.XGBClassifier()
    xgb_dist_angle = xgb1.fit(x_train, y_train)
    y_pred = xgb_dist_angle.predict(x_val)

    # Calculate score
    xgb_dist_angle_score = xgb_dist_angle.score(x_val, y_val)
    print('XGBoost score on distance and angle: ', xgb_dist_angle_score)

    # Classification report
    xgb_dist_angle_report = classification_report(y_val, y_pred)
    print('XGBoost classification report on distance and angle: ', xgb_dist_angle_report)

    # Save the model in joblib format
    joblib.dump(xgb_dist_angle, 'xgb1.joblib')
    # # Confusion matrix
    # xgb_dist_angle_confusion = confusion_matrix(y_val, y_pred)
    # print('XGBoost confusion matrix on distance and angle: ', xgb_dist_angle_confusion)

    return xgb_dist_angle, x_val,y_val

def plot_roc_curve(model,x_val,y_val):
    # ROC curve
    y_pred_proba = model.predict_proba(x_val)[:,1]
    xgb_dist_angle_roc_auc = roc_auc_score(y_val, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label='XGBoost (area = %0.2f)' % xgb_dist_angle_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    plt.show()

# Shot probability model percentile based on distance and angle
def shot_probability(model, x_val):
    xgb_probability = model.predict_proba(x_val)[:,1]
    x_axis = np.arange(len(xgb_probability))[::-1]*100/len(xgb_probability)
    y_axis = np.sort(xgb_probability)[::-1]
    plt.plot(x_axis, y_axis)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Goal Rate')
    plt.title('Shot probability model percentile based on distance and angle')
    plt.show()

def cumulative_probability(model, x_val):
    xgb_probability = model.predict_proba(x_val)[:,1]
    x_axis = np.arange(len(xgb_probability))[::-1]*100/len(xgb_probability)
    y_axis = np.sort(xgb_probability)[::-1]
    y_axis_cum = np.cumsum(y_axis)/np.sum(y_axis)
    plt.plot(x_axis, y_axis_cum)
    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Cumulative proportion of goals')
    plt.title('Cumulative probability of goals')
    plt.show()

# Calibration plot using CalibrationDisplay
def calibration_plot(model, x_val, y_val):
    CalibrationDisplay.from_estimator(model, x_val, y_val, n_bins=10, name='XGBoost', ax=None)
    plt.title('Reliability Diagram')
    plt.show()

def confusion_matrix(model, x_val, y_val):
    y_pred = model.predict(x_val)
    confusion = confusion_matrix(y_val, y_pred)
    print('Confusion matrix: ', confusion)

def comet(path):
    df = load_data(path)
    df = df[['game_id','period','coordinates_x','coordinates_y','shot_distance','shot_angle','goal','is_empty_net']]
    analyze_data(df)
    # Split the data into train and test 
    X = df[['shot_distance','shot_angle']].astype(np.float32)
    y = df['goal']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    xgb1, x_val1, y_val1 = XGBoost1(x_train, x_val, y_train, y_val)
    plot_roc_curve(xgb1, x_val1, y_val1)
    shot_probability(xgb1, x_val1)
    cumulative_probability(xgb1, x_val1)
    calibration_plot(xgb1, x_val1, y_val1)
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='ift-6758-team-8',
        workspace="Rachel98",
    )
    # load the model
    model1 = joblib.load('xgb1.joblib')
    experiment.log_dataset_hash(df)
    experiment.log_parameters({'model': 'XGBoost', 'features': 'distance and angle'})
    experiment.log_metrics({'score': accuracy_score(y_val1, xgb1.predict(x_val1))})
    experiment.log_metrics({'roc_auc_score': roc_auc_score(y_val1, xgb1.predict_proba(x_val1)[:,1])})
    experiment.log_metrics({'precision_score': precision_score(y_val1, xgb1.predict(x_val1))})
    experiment.log_metrics({'recall_score': recall_score(y_val1, xgb1.predict(x_val1))})
    experiment.log_metrics({'f1_score': f1_score(y_val1, xgb1.predict(x_val1))})
    experiment.log_metrics({'accuracy_score': accuracy_score(y_val1, xgb1.predict(x_val1))})
    # experiment.log_model(model1, 'xgb1.joblib')
    experiment.end()

if __name__ == '__main__':
    comet('D:/NHLPro/data/df_all_season.csv')