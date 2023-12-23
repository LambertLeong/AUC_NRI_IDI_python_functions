import numpy as np
import pytest, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import bootstrap_results
from sklearn import metrics

def test_bootstrap_results_with_valid_data():
    # Define test data with at least two classes
    y_truth = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    num_bootstraps = 10  # Reduced number for test speed

    # Call the function under test
    base_thresh, mean_tprs, mean_fprs, tprs, fprs = bootstrap_results(y_truth, y_pred, num_bootstraps)

    # Assertions
    assert len(base_thresh) == 101  # Check base threshold length
    assert len(mean_tprs) == 101  # Check mean TPRs length
    assert len(mean_fprs) == 101  # Check mean FPRs length
    assert len(tprs) == num_bootstraps  # Check number of TPR arrays
    assert len(fprs) == num_bootstraps  # Check number of FPR arrays

def test_bootstrap_results_error_on_same_values():
    # Define test data with same values
    y_truth = np.array([1, 1, 1, 1])
    y_pred = np.array([0.2, 0.5, 0.6, 0.8])

    # Expect ValueError when all y_truth values are the same
    with pytest.raises(ValueError):
        bootstrap_results(y_truth, y_pred)

def test_bootstrap_results_zero_bootstraps():
    y_truth = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    num_bootstraps = 0

    base_thresh, mean_tprs, mean_fprs, tprs, fprs = bootstrap_results(y_truth, y_pred, num_bootstraps)
    # Assertions based on expected behavior

def test_bootstrap_results_high_bootstraps():
    y_truth = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    num_bootstraps = 10000

    base_thresh, mean_tprs, mean_fprs, tprs, fprs = bootstrap_results(y_truth, y_pred, num_bootstraps)
    # Assertions based on expected behavior

def test_bootstrap_results_empty_arrays():
    y_truth = np.array([])
    y_pred = np.array([])
    num_bootstraps = 1000

    with pytest.raises(ValueError):
        bootstrap_results(y_truth, y_pred, num_bootstraps)

def test_bootstrap_results_invalid_data():
    y_truth = "invalid data"
    y_pred = "invalid data"
    num_bootstraps = 1000

    with pytest.raises(TypeError):
        bootstrap_results(y_truth, y_pred, num_bootstraps)
