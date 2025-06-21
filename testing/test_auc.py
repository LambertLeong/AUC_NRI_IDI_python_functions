import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import calculate_auc_with_ci, auc_difference_pvalue
from sklearn import metrics

def test_calculate_auc_with_ci_valid_data():
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

def test_auc_difference_pvalue_range():
    truth = np.array([0, 1, 0, 1])
    ref = np.array([0.1, 0.4, 0.35, 0.8])
    new = np.array([0.2, 0.5, 0.3, 0.9])

    diff, pval = auc_difference_pvalue(truth, ref, new, num_bootstraps=50)
    assert isinstance(diff, float)
    assert isinstance(pval, float)
    assert 0 <= pval <= 1

def test_auc_difference_pvalue_identical_models():
    truth = np.array([0, 1, 0, 1])
    preds = np.array([0.1, 0.4, 0.35, 0.8])

    diff, pval = auc_difference_pvalue(truth, preds, preds, num_bootstraps=20)
    assert diff == pytest.approx(0.0)
    assert pval == pytest.approx(1.0)

def test_auc_difference_pvalue_significant_difference():
    y_truth = np.array([0, 0, 1, 1])
    ref = np.array([0.5, 0.5, 0.5, 0.5])
    new = np.array([0.0, 0.0, 1.0, 1.0])

    diff, pval = auc_difference_pvalue(y_truth, ref, new, num_bootstraps=20)
    assert diff > 0.4
    assert pval < 0.01

def test_auc_difference_pvalue_invalid_inputs():
    truth = np.array([0, 1])
    ref = np.array([0.1, 0.4])
    new = np.array([0.2])

    with pytest.raises(ValueError):
        auc_difference_pvalue(truth, ref, new)

    with pytest.raises(TypeError):
        auc_difference_pvalue([0, 1], [0.1, 0.4], [0.2, 0.3])

def test_auc_difference_pvalue_reproducible():
    truth = np.array([0, 1, 0, 1])
    ref = np.array([0.1, 0.4, 0.35, 0.8])
    new = np.array([0.2, 0.5, 0.3, 0.9])

    diff1, pval1 = auc_difference_pvalue(truth, ref, new, num_bootstraps=30)
    diff2, pval2 = auc_difference_pvalue(truth, ref, new, num_bootstraps=30)

    assert diff1 == pytest.approx(diff2)
    assert pval1 == pytest.approx(pval2)

