import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models):
        self.base_models = base_models

    def fit(self, X, y):
        self.base_models_ = [clone(model) for model in self.base_models]
        for model in self.base_models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.base_models_])
        majority_vote = mode(predictions, axis=1)[0].flatten()
        return majority_vote, predictions