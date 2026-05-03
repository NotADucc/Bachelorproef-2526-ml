import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .common import evaluate_model


def run_isolation_forest(X, y, df, model_name, random_state, params):
    model = IsolationForest(**params, random_state=random_state)
    labels = model.fit_predict(X)
    score = model.decision_function(X)
    return evaluate_model(
        df=df,
        labels=labels,
        score=score,
        y=y,
        model_name=model_name,
        params=params
    )


def run_lof(X, y, df, model_name, random_state, params):
    model = LocalOutlierFactor(**params)

    if params.get("novelty", False):
        model.fit(X)
        labels = model.predict(X)
        score = model.decision_function(X)
    else:
        labels = model.fit_predict(X)
        score = model.negative_outlier_factor_

    return evaluate_model(
        df=df,
        labels=labels,
        score=score,
        y=y,
        model_name=model_name,
        params=params
    )


def run_ocsvm(X, y, df, model_name, random_state, params):
    model = OneClassSVM(**params)
    model.fit(X)
    labels = model.predict(X)
    score = model.decision_function(X)
    return evaluate_model(
        df=df,
        labels=labels,
        score=score,
        y=y,
        model_name=model_name,
        params=params
    )


def dbscan_scores(X, labels):
    core_mask = labels != -1

    if core_mask.sum() == 0:
        return np.zeros(len(X))

    nbrs = NearestNeighbors(n_neighbors=1).fit(X[core_mask])
    distances, _ = nbrs.kneighbors(X)

    return distances.ravel()


def run_dbscan(X, y, df, model_name, random_state, params):
    model = DBSCAN(**params)
    labels = model.fit_predict(X)
    score = dbscan_scores(X, labels)
    return evaluate_model(
        df=df,
        labels=labels,
        score=score,
        y=y,
        model_name=model_name,
        params=params
    )


def run_elliptic(X, y, df, model_name, random_state, params):
    model = EllipticEnvelope(
        **params,
        random_state=random_state
    )
    model.fit(X)
    labels = model.predict(X)
    score = model.decision_function(X)
    return evaluate_model(
        df=df,
        labels=labels,
        score=score,
        y=y,
        model_name=model_name,
        params=params
    )
