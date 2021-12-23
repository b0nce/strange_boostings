import lightgbm as lgb
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class LGBMLinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lightgbm_model=None, linear_model=None, random_state=None):
        assert lightgbm_model is None or isinstance(
            lightgbm_model, lgb.LGBMClassifier
        ), f"Only LGBMClassifier isntances are allowed!"
        self.lightgbm_model = lightgbm_model
        self.linear_model = linear_model
        self.random_state = random_state

    def _validate_data(self, X, y=None, reset=True, **check_array_params):
        if hasattr(self, "n_features_in_"):
            if self.n_features_in_ != check_array(X).shape[1]:
                raise ValueError(f"Input has incorrect shape: {X.shape}")
        else:
            self.n_features_in_ = check_array(X).shape[1]
        if y is not None:
            return check_X_y(X, y)
        else:
            return check_array(X)

    def fit(self, X, y, valset=None):
        X, y = self._validate_data(X, y)
        if valset is not None:
            valset = self._validate_data(*valset)
        self.classes_ = unique_labels(y)

        if self.lightgbm_model is not None:
            self.lightgbm_model_ = clone(self.lightgbm_model)
        else:
            self.lightgbm_model_ = lgb.LGBMClassifier(
                boosting_type="dart", class_weight="balanced", random_state=self.random_state
            )
        if self.linear_model is not None:
            self.linear_model_ = clone(self.linear_model)
        else:
            self.linear_model_ = SGDClassifier(loss="modified_huber", random_state=self.random_state)

        self.lightgbm_model_.fit(X, y, eval_set=valset, verbose=False)
        X_contrib = self.lightgbm_model_.booster_.predict(X, pred_contrib=True)
        self.linear_model_.fit(X_contrib, y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        return self.linear_model_.predict(self.lightgbm_model_.booster_.predict(X, pred_contrib=True))

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        try:
            return self.linear_model_.predict_proba(
                self.pca_.transform(self.lightgbm_model_.booster_.predict(X, pred_contrib=True))
            )
        except AttributeError:
            pred = self.linear_model_.decision_function(
                self.pca_.transform(self.lightgbm_model_.booster_.predict(X, pred_contrib=True))
            )
            return np.repeat([[-1, 0]], len(pred), axis=0) + sigmoid(pred)[:, None]
