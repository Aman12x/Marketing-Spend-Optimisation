#!/usr/bin/env python

import pandas as pd
from lightgbm import LGBMClassifier
from pandas import factorize
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay, PrecisionRecallDisplay

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_models(X, y):
    # Model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=300)
    xgb = XGBClassifier(
        n_estimators=300,
        objective="binary:logistic",
        tree_method="hist",
        eta=0.1,
        max_depth=3,
    )
    lgb = LGBMClassifier(n_estimators=300)

    rf.fit(X_train, y_train)
    print("Random Forest trained")
    xgb.fit(X_train, y_train)
    print("XGB trained")
    lgb.fit(X_train, y_train)
    print("LGB trained")

    return rf, xgb, lgb, X_test, y_test


from sklearn.metrics import accuracy_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt


def evaluate_model(model_name, model, pred, X_test, y_test):
    print("Accuracy of %s: " % model_name, accuracy_score(y_test, pred))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.show()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()


def save_models(rf, xgb, lgb):
    # Save models
    pickle.dump(rf, open("../models/random_forest.model", "wb"))
    pickle.dump(xgb, open("../models/xgboost.model", "wb"))
    pickle.dump(lgb, open("../models/lightgbm.model", "wb"))
