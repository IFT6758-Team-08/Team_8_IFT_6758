from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from AdvancedModels_XGBoost1 import *
from comet_ml import Experiment
import joblib


# Data Preprocessing
def preprocess_data(df):
    # Convert rebound to 0 and 1 instead of True and False
    df['rebound'] = df['rebound'].astype(int)
    # Rearranging columns to make it easier to process data
    df=df[['game_id','period','period_time','coordinates_x','coordinates_y','shot_distance','shot_angle','secondary_type','last_event_type','time_from_last_event(s)','distance_from_last_event','rebound','angle_change','speed','last_event_coordinates_x','last_event_coordinates_y','goal']]
    # Convert period_time to seconds
    df['period_time'] = df['period_time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    # Convert secondary_type and last_event_type to numeric
    # df['secondary_type'] = df['secondary_type'].astype('category').cat.codes
    # df['last_event_type'] = df['last_event_type'].astype('category').cat.codes
    # Convert secondary_type and last_event_type to dummy variables
    df = pd.get_dummies(df, columns=['secondary_type','last_event_type'], drop_first=True)
    return df

# XGBoost on all features except goal
def XGBoost2(X_train, X_test, y_train, y_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Hyper parameter tuning
    # Create the parameter grid based on the results of random search
    param_grid = {
        'max_depth': [3, 4, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5,1,1.5,2,5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 500, 1000],

    }

    # Create a based model
    xgb = XGBClassifier(n_estimators=1000, objective='binary:logistic', silent=True, nthread=1)
    stratifiedSearch = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    randomSearch = RandomizedSearchCV(estimator = xgb, param_distributions = param_grid, n_iter = 3, cv = stratifiedSearch, verbose=3, random_state=42, n_jobs = 4)
    
    # Fit the random search model
    randomSearch.fit(X_train, y_train)

    # Print best parameters and best score from random search 
    # print("Best parameters from random search: ", randomSearch.best_params_)
    # print("Best score from random search: ", randomSearch.best_score_)
    # print("Best estimator from random search: ", randomSearch.best_estimator_)
    # print("Best index from random search: ", randomSearch.best_index_)
    # Store best parameters in a variable
    best_params = randomSearch.best_params_
    best_score = randomSearch.best_score_
    best_estimator = randomSearch.best_estimator_
    best_index = randomSearch.best_index_
    return best_params, best_score, best_estimator, best_index

def XGB_with_best_estimator_all_feature(X_train, X_test, y_train, y_test):
    # Training the XGBoost model on the Training set with best parameters from random search

    # Fit the model on the trainng data
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.6,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=1.5, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.05, max_bin=256,
              max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
              max_depth=7, max_leaves=0, min_child_weight=10,
              monotone_constraints='()', n_estimators=500, n_jobs=1, nthread=1,
              num_parallel_tree=1, predictor='auto')

    xgb.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = xgb.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    # cm = confusion_matrix(y_test, y_pred)
    # print("The confusion matrix is : ",cm)

    # Accuracy
    print("The accuracy score is: ",accuracy_score(y_test, y_pred))

    # Making classification report 
    from sklearn.metrics import classification_report
    print("The classification report: ",classification_report(y_test, y_pred))

    # save the model to joblib
    joblib.dump(xgb, 'xgb2.joblib')
    return xgb, X_test, y_test

def plot_roc_curve1(model,x_val,y_val, model_name):
    # ROC curve
    y_pred_proba = model.predict_proba(x_val)[:,1]
    xgb_dist_angle_roc_auc = roc_auc_score(y_val, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=model_name+' (area = %0.2f)' % xgb_dist_angle_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    plt.show()


def comet(path):
    df = load_data(path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(df.drop('goal', axis=1), df['goal'], test_size=0.2, random_state=42)
    params, score,estimator,index = XGBoost2(X_train, X_test, y_train, y_test)
    xgb2, X_test2, y_test2 = XGB_with_best_estimator_all_feature(X_train, X_test, y_train, y_test)
    plot_roc_curve1(xgb2,X_test2,y_test2,"XGBoost_all_features")
    shot_probability(xgb2, X_test2)
    cumulative_probability(xgb2, X_test2)
    calibration_plot(xgb2, X_test2, y_test2)

    # Log the experiment and the parameters to comet.ml
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='ift-6758-team-8',
        workspace="Rachel98",
    )
    xgb2 = joblib.load('xgb2.joblib')
    experiment.log_parameters(params)
    experiment.log_metric("best_score", score)
    experiment.log_metric("best_index", index)
    experiment.log_metric("best_estimator", estimator)
    # Log the metrics
    experiment.log_metric("accuracy", accuracy_score(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("precision", precision_score(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("recall", recall_score(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("f1", f1_score(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("roc_auc", roc_auc_score(y_test2, xgb2.predict(X_test2)))
    # experiment.log_metric("confusion_matrix", confusion_matrix(y_test2, xgb2.predict(X_test2)))
    experiment.log_metric("classification_report", classification_report(y_test2, xgb2.predict(X_test2)))
    # Log the model
    experiment.log_model(xgb2,'xgb2.joblib')
    experiment.end()

if __name__ == "__main__":
    comet("D:/NHlPro/data/M2_added_features_all.csv")