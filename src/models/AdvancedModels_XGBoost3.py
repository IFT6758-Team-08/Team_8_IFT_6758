from xgboost import plot_importance
from AdvancedModels_XGBoost2 import *
from comet_ml import Experiment
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

def plot_feature_importance(model):
    # plot feature importance
    plot_importance(model)
    plt.show()

def correlation(df):
    # Create heatmap to show correlation between features
    corr_matrix = df[['shot_angle','speed','shot_distance','distance_from_last_event','period_time','coordinates_y','time_from_last_event(s)']].corr().round(2)

    # Heatmap
    plt.figure(figsize=(10,10))
    sns.heatmap(data=corr_matrix, annot=True)
    plt.show()

def explain_model_prediction(model, X_test):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # Calculate Shap values
    shap_values = explainer.shap_values(X_test)

    # Make plot. Index of [1] is explained in text below.
    shap.summary_plot(shap_values, X_test)

# Lets take 7 most important features and train the model again
def XGBoost3(X_train,y_train):
    
    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid = {"max_depth": [3, 5, 7],
                    "learning_rate" : [0.01, 0.05, 0.1],
                    "gamma": [0.5, 1, 1.5],
                    "min_child_weight": [1, 5, 10],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "subsample": [0.6, 0.8, 1.0],
                    "reg_alpha": [0, 0.5, 1],
                    "reg_lambda": [1, 1.5, 2],
                    "scale_pos_weight": [1, 3, 5]}

    # Create a based model
    xgb3 = XGBClassifier(n_estimators=1000, objective='binary:logistic', silent=True, nthread=1)
    stratifiedSearch = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    randomSearch = RandomizedSearchCV(estimator = xgb3, param_distributions = param_grid, n_iter = 3, cv = stratifiedSearch, verbose=3, random_state=42, n_jobs = 4)
    
    # Fit the random search model
    randomSearch.fit(X_train, y_train)

    # Save the model in joblib format
    joblib.dump(randomSearch, 'XGBoost3.joblib')

    # Print best parameters and best score from random search 
    # print("Best parameters from random search: ", randomSearch.best_params_)
    # print("Best score from random search: ", randomSearch.best_score_)
    # print("Best estimator from random search: ", randomSearch.best_estimator_)
    # print("Best index from random search: ", randomSearch.best_index_)
    best_params = randomSearch.best_params_
    best_score = randomSearch.best_score_
    best_estimator = randomSearch.best_estimator_
    best_index = randomSearch.best_index_
    return best_params, best_score, best_estimator, best_index

# Training the XGBoost model on the Training set with best parameters from random search
def XGB_with_best_estimator_7_feature(X_train, X_test, y_train, y_test):
    
    xgb3 = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1.0,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=1, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.01, max_bin=256,
              max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
              max_depth=7, max_leaves=0, min_child_weight=1,
              monotone_constraints='()', n_estimators=1000, n_jobs=1, nthread=1,
              num_parallel_tree=1, predictor='auto')


    xgb3.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = xgb3.predict(X_test)

    # Making the Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion matrix: ", cm)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score: ", accuracy)

    # Making classification report 
    print("The classification report: ",classification_report(y_test, y_pred))

    # Save the model in joblib format
    joblib.dump(xgb3, 'xgb3.joblib')

    return xgb3, X_test, y_test


def comet(path):
    df = load_data(path)
    # analyze_data(df)
    df = preprocess_data(df)
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['goal'],axis=1), df['goal'], test_size=0.2, random_state=42)
    xgb2, X_test2, y_test2 = XGB_with_best_estimator_all_feature(X_train, X_test, y_train, y_test)
    plot_feature_importance(xgb2)
    correlation(df)
    explain_model_prediction(xgb2, X_test2)
    best_params, best_score, best_estimator, best_index = XGBoost3(X_train, y_train)
    df = df[['shot_angle','speed','shot_distance','distance_from_last_event','period_time','coordinates_y','time_from_last_event(s)','goal']]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['goal'],axis=1), df['goal'], test_size=0.2, random_state=42)
    xgb3, X_test3, y_test3 = XGB_with_best_estimator_7_feature(X_train, X_test, y_train, y_test)
    plot_roc_curve1(xgb3, X_test3, y_test3,"XGBoost_best_feature")
    shot_probability(xgb3, X_test3)
    cumulative_probability(xgb3, X_test3)
    calibration_plot(xgb3, X_test3, y_test3)

    # Log the experiment and the parameters to comet.ml
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='ift-6758-team-8',
        workspace="Rachel98",
    )
    xgb2 = joblib.load('xgb3.joblib')
    experiment.log_parameters(best_params)
    experiment.log_metric("best_score", best_score)
    experiment.log_metric("best_index", best_index)
    experiment.log_metric("best_estimator", best_estimator)
    # Log the metrics
    experiment.log_metric("accuracy", accuracy_score(y_test3, xgb3.predict(X_test3)))
    experiment.log_metric("precision", precision_score(y_test3, xgb3.predict(X_test3)))
    experiment.log_metric("recall", recall_score(y_test3, xgb3.predict(X_test3)))
    experiment.log_metric("f1", f1_score(y_test3, xgb3.predict(X_test3)))
    experiment.log_metric("roc_auc", roc_auc_score(y_test3, xgb3.predict(X_test3)))
    # experiment.log_metric("confusion_matrix", confusion_matrix(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("classification_report", classification_report(y_test3, xgb3.predict(X_test3)))
    # Log the model
    experiment.log_model(xgb3,'xgb3.joblib')
    experiment.end()

if __name__ == "__main__":
    comet("D:/NHlPro/data/M2_added_features_all.csv")