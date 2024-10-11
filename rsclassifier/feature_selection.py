from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def feature_selection_using_random_forest(X, y, k):
    rfc = RandomForestClassifier(random_state = 42)
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    feature_names = X.columns
    feature_importances = {}
    for i in range(len(feature_names)):
        feature_importances[feature_names[i]] = importances[i]
    ranked_features = list((dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)).keys()))
    return ranked_features[:k]

def feature_selection_using_decision_tree(X, y, k):
    dt = DecisionTreeClassifier(random_state = 42)
    dt.fit(X, y)
    importances = dt.feature_importances_
    feature_names = X.columns
    feature_importances = {}
    for i in range(len(feature_names)):
        feature_importances[feature_names[i]] = importances[i]
    ranked_features = list((dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)).keys()))
    return ranked_features[:k]