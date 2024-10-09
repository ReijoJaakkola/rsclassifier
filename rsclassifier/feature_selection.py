import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from rsclassifier.information_theory import information
from queue import Queue
from tqdm import tqdm

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
    
def feature_selection_using_information_gain(X, y, k):
    target = y.name
    Z = pd.concat([X,y], axis = 1)
    information_upper_bound = np.log2(len(Z[target].unique())) + 1
    
    features = []
    q = Queue()
    q.put(Z)
    for _ in tqdm(range(k), desc = 'Selecting features...'):
        Z = q.get()
        N = len(Z)

        best_feature = None
        smallest_information_value = information_upper_bound
        left = None
        right = None
        left_is_more_diverse = None
        current_features = Z.columns.difference([target] + features)
        for feature in current_features:
            Z1 = Z[Z[feature] == 1].copy()
            Z1.drop(columns = [feature], inplace = True)
            Z2 = Z[Z[feature] == 0].copy()
            Z2.drop(columns = [feature], inplace = True)
            n1 = len(Z1)
            n2 = len(Z2)
            I1 = information(Z1[target])
            I2 = information(Z2[target])

            # TODO: Handle the case where n1 = 0 or n2 = 0.
            # TODO: Early stopping? When do we stop splitting?

            information_value = (n1 / N) * I1 + (n2 / N) * I2
            if information_value <= smallest_information_value:
                best_feature = feature
                smallest_information_value = information_value
                left = Z1
                right = Z2
                left_is_more_diverse = I1 > I2

        features.append(best_feature)
        if left_is_more_diverse:
            q.put(left)
            q.put(right)
        else:
            q.put(left)
            q.put(right)
    return features