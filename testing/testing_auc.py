import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import calculate_auc_with_ci
from sklearn import metrics

def test_calculate_auc_with_ci_valid_data(mocker):
    # Mock the bootstrap_results function

    # Define test data
    ground_truth = np.array([0, 1, 0, 1])
    predicted_probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    
    # Call the function under test
    result = calculate_auc_with_ci(ground_truth, predicted_probabilities)

    # Assertions
    assert len(result) == 6  # Check if the function returns the correct number of elements
    assert isinstance(result[4], float)  # Mean AUC should be a float
    assert 0 <= result[4] <= 1  # Mean AUC should be between 0 and 1

def test_true_positive_rate_bounds():
    ground_truth = np.array([0, 1, 0, 1])
    predicted_probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    num_bootstraps = 100

    _, mean_tprs, lower_tprs, upper_tprs, _, _ = calculate_auc_with_ci(ground_truth, predicted_probabilities, num_bootstraps)
    
    assert all(lower_tprs <= upper_tprs), "Lower true positive rates should be less than or equal to upper true positive rates"

def test_mean_auc_and_std_dev():
    ground_truth = np.array([0, 1, 1, 0])
    predicted_probabilities = np.array([0.2, 0.6, 0.8, 0.3])
    num_bootstraps = 100

    _, _, _, _, mean_auc, std_auc = calculate_auc_with_ci(ground_truth, predicted_probabilities, num_bootstraps)

    assert 0 <= mean_auc <= 1, "Mean AUC should be between 0 and 1"
    assert 0 <= std_auc, "Standard deviation of AUC should be non-negative"

def test_empty_input():
    with pytest.raises(ValueError):
        calculate_auc_with_ci(np.array([]), np.array([]), 1000)

def test_non_array_input():
    with pytest.raises(TypeError):
        calculate_auc_with_ci([0, 1, 1, 0], [0.2, 0.6, 0.8, 0.3], 1000)

def test_invalid_ground_truth():
    with pytest.raises(ValueError):
        calculate_auc_with_ci(np.array([1, 1, 1, 1]), np.array([0.2, 0.6, 0.8, 0.3]), 1000)

