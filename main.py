import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import xgboost

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Mulanjoe\BOT PROJECT\archive\EURUSD_15m_BID_01.01.2010-31.12.2016.csv")

from data_inspection import inspect_data
#inspect_data(df)

from check_safe_removal_of_duplicates import is_removing_duplicates_safe
#is_removing_duplicates_safe(df)

from remove_duplicates import remove_duplicates
df = remove_duplicates(df)

from feature_engineering import create_features
df = create_features(df)

#print(df[["Time",
 #         "Close",
  #        "return_1",
   #       "ma_5",
    #      "ma_20",
     #     "volatility_20",
      #    "price_vs_ma20"]].tail(10))

from create_target import create_target
df = create_target(df)

#print(df[["Time",
 #         "Close",
  #        "next_close",
   #       "target"]].tail(10))

#print(df["target"].value_counts())

#print("Class percentages\n:")
#print(df["target"].value_counts(normalize=True) * 100) #Check if classes are balanced

df = df.dropna().copy()
#print(df.isnull().sum())
#print(df.shape)

#print(df.columns)

###Select features
features = ['Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'return_1',
            'ma_5',
            'ma_20',
            'volatility_20',
            'price_vs_ma20',]

X = df[features]
y = df["target"]

#print("Feature's shape", X.shape)
#print("Target's shape", y.shape)

###Splitting data chronologically
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

def check_split_data(X_train, X_test, y_train, y_test):
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

#check_split_data(X_train, X_test, y_train, y_test)


##MODELLING(CFT)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

trading_model = XGBClassifier(
    n_estimators = 100,
    max_depth = 4,
    learning_rate = 0.1,
    random_state = 42,
    eval_metric = "logloss"
)

##Train the model
trading_model.fit(X_train, y_train)

##Make predictions on unseen data
predictions = trading_model.predict(X_test)

##Evaluate predictions
accuracy_score = accuracy_score(predictions, y_test)
#print("Accuracy_score:", accuracy_score)

from inspect_model_mistakes import *
#inspect_mistakes(y_test, predictions)

##############EVALUATION IN PROGRESS###################################

#####Checking for useful features i.e. rank the 10 input features by their importance
importance = pd.Series(
    trading_model.feature_importances_,
    index = X_train.columns
).sort_values(ascending=False)

#print(importance)

def comparing(X_train,X_test, y_train ,y_test):
    ###COMPARE 2 MODELS; 1 WITH VOLUME, THE OTHER WITHOUT
    #MODEL A
    from sklearn.metrics import accuracy_score
    model_A = XGBClassifier(
        n_estimators = 100,
        max_depth = 4,
        learning_rate = 0.1,
        random_state = 42,
        eval_metric = "logloss"
    )

    model_A.fit(X_train, y_train)
    pred_from_A = model_A.predict(X_test)

    #MODEL B
    features_no_volume = [col for col in features if col!="Volume"]

    X_train_no_volume = X_train[features_no_volume]
    X_test_no_volume = X_test[features_no_volume]

    print("Features in model A", list(X_train.columns))
    print("Number of model A features", X_train.shape[1])

    print("Features in model B", list(X_train_no_volume.columns))
    print("Number of model B features", X_train_no_volume.shape[1])

    model_B = XGBClassifier(
        n_estimators = 100,
        max_depth = 4,
        learning_rate = 0.1,
        random_state = 42,
        eval_metric = "logloss"
    )

    model_B.fit(X_train_no_volume, y_train)
    pred_from_B = model_B.predict(X_test_no_volume)

    ##Compare
    print("MODEL A: WITH VOLUME")
    print("Accuracy:",  accuracy_score(pred_from_A, y_test))
    print(classification_report(y_test, pred_from_A))

    print("MODEL B: WITHOUT VOLUME")
    print("Accuracy:", accuracy_score(pred_from_B, y_test))
    print(classification_report(y_test, pred_from_B))

##comparing(X_train, X_test, y_train, y_test)

###Split the training data into more training and validation coz we are experimeting with parameters
validation_index = int(len(X_train) * 0.8)

X_train_fit = X_train.iloc[:validation_index]
X_val = X_train.iloc[validation_index:]

y_train_fit = y_train.iloc[:validation_index]
y_val = y_train.iloc[validation_index:]

def check_split_experimenting_data(X_train_fit, X_val,X_test):
    print("Training:", X_train_fit.shape)
    print("Validation:", X_val.shape)
    print("Final test:", X_test.shape)

#check_split_experimenting_data(X_train_fit, X_val,X_test)

##Experiment with XGBoost parameters

def experiment_with_parameters(X_train_fit, y_train_fit,y_val):
    from sklearn.metrics import accuracy_score, f1_score
    configs = [
        {"n_estimators": 100, "max_depth":3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth":4, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth":3, "learning_rate": 0.05}
    ]

    for i, params in enumerate(configs, start=1):
        experiment_model = XGBClassifier(
            **params,
            random_state= 42,
            eval_metric= "logloss"
        )

        experiment_model.fit(X_train_fit, y_train_fit)
        val_pred = experiment_model.predict(X_val)

        accuracy = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred)

        print(f"\nConfiguration{i}:{params}")
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Class 1 F1-score: {f1:.4f}")

#experiment_with_parameters(X_train_fit, y_train_fit,y_val)

###Config 2 had the best f1_score. So we will use it to model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Train using config 2 parameters
best_model = XGBClassifier(
    n_estimators= 100,
    max_depth= 4,
    learning_rate= 0.1,
    random_state= 42,
    eval_metric= "logloss"
)

best_model.fit(X_train_fit, y_train_fit)

val_pred = best_model.predict(X_val)

def results_from_best_model(y_val, val_pred):
    print("****** VALIDATION CONFUSION MATRIX *******")
    print(confusion_matrix(y_val, val_pred))

    print("***** VALIDATION CLASSIFICATION REPORT *****")
    print(classification_report(y_val, val_pred))

#results_from_best_model

###FINAL TEST EVALUATION
from sklearn.metrics import accuracy_score, classification_report
test_pred = best_model.predict(X_test)

print("FINAL TEST ACCURACY:")
print(accuracy_score(y_test, test_pred))

print("\nFINAL CLASSIFICATION REPORT:")
print(classification_report(y_test, test_pred))
