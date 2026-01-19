import numpy as np
import pytest
from glum import GeneralizedLinearRegressor

@pytest.fixture
def linear_data():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    coef = np.array([1.0, -2.0, 0.5])
    y = X @ coef + np.random.randn(50) * 0.1
    return X, y, coef

@pytest.fixture
def logistic_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    logits = X @ np.array([2.0, -1.0])
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    return X, y

# -----------------------
# Linear regression tests
# -----------------------
def test_linear_regression_fit(linear_data):
    X, y, coef = linear_data
    model = GeneralizedLinearRegressor(family="gaussian", l1_ratio=0.0, alpha=0.0)
    model.fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, rtol=0.2)

def test_linear_regression_predict(linear_data):
    X, y, _ = linear_data
    model = GeneralizedLinearRegressor(family="gaussian", l1_ratio=0.0, alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape

# -----------------------
# Logistic regression tests
# -----------------------
def test_logistic_regression_fit(logistic_data):
    X, y = logistic_data
    model = GeneralizedLinearRegressor(family="binomial", l1_ratio=0.0, alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds >= 0) and np.all(preds <= 1)

def test_logistic_regression_predict_class(logistic_data):
    X, y = logistic_data
    model = GeneralizedLinearRegressor(family="binomial", l1_ratio=0.0, alpha=0.0)
    model.fit(X, y)
    preds_class = (model.predict(X) > 0.5).astype(int)
    assert set(np.unique(preds_class)).issubset({0, 1})

# -----------------------
# Regularization tests
# -----------------------
def test_l1_regularization(linear_data):
    X, y, _ = linear_data
    model = GeneralizedLinearRegressor(family="gaussian", l1_ratio=1.0, alpha=0.5)
    model.fit(X, y)
    # L1 should shrink coefficients
    assert np.sum(np.abs(model.coef_)) <= np.sum(np.abs(np.linalg.lstsq(X, y, rcond=None)[0]))

def test_l2_regularization(linear_data):
    X, y, _ = linear_data
    model = GeneralizedLinearRegressor(family="gaussian", l1_ratio=0.0, alpha=0.5)
    model.fit(X, y)
    # L2 should shrink coefficients (ridge)
    assert np.linalg.norm(model.coef_) <= np.linalg.norm(np.linalg.lstsq(X, y, rcond=None)[0])

# -----------------------
# Exception tests
# -----------------------
def test_fit_before_predict_raises(linear_data):
    X, y, _ = linear_data
    model = GeneralizedLinearRegressor(family="gaussian")
    # predict before fit should raise AttributeError
    with pytest.raises(AttributeError):
        model.predict(X)