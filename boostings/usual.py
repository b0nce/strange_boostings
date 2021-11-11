import numpy as np
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SimpleBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, n_estimators=10, l2_reg=0):
        self.coef_ = np.ones(n_estimators + 1) / (n_estimators + 1)
        self.regrs = []
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.l2_reg = l2_reg

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.regrs.append(DummyRegressor().fit(X, y))
        preds = [self.regrs[-1].predict(X)]

        res = y - preds[-1]

        for idx in range(len(self.coef_) - 1):
            self.regrs.append(clone(self.base_estimator).fit(X, res))
            preds.append(self.regrs[-1].predict(X))
            res = res - self.coef_[idx] * preds[-1]
        b = np.stack(preds, axis=0).T
        self.coef_ = np.linalg.pinv(b.T @ b + np.eye(len(b.T)) * self.l2_reg) @ b.T @ y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return np.sum([regr.predict(X) * c for regr, c in zip(self.regrs, self.coef_)], axis=0)


class NestedBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, layers_n_estimators=(2, 2, 2, 2, 2), layers_regs=(0.1, 0.05, 0.01, 0.001, 0)):
        assert len(layers_n_estimators) == len(
            layers_regs
        ), "layers_n_estimators and layers_regs must be the same length!"
        self.boosting = clone(base_estimator)
        self.layers_n_estimators = layers_n_estimators
        self.layers_regs = layers_regs
        for el, reg in zip(layers_n_estimators, layers_regs):
            self.boosting = SimpleBoostingRegressor(self.boosting, n_estimators=el, l2_reg=reg)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.boosting.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return self.boosting.predict(X)
