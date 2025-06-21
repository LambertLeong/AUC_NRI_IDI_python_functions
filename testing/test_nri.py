import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import calculate_nri, category_free_nri, check_cat, track_movement, nri_pvalue

def test_track_movement():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    new = np.array([0.2, 0.1, 0.4, 0.3])
    indices = [0, 1, 2, 3]

    expected_up, expected_down = 2, 2

    up, down = track_movement(ref, new, indices)

    assert up == expected_up
    assert down == expected_down

def test_check_cat():
    thresholds = [0.2, 0.5, 0.8]

    # Test with values within the threshold range
    assert check_cat(0.1, thresholds) == 0
    assert check_cat(0.3, thresholds) == 1
    assert check_cat(0.6, thresholds) == 2
    assert check_cat(0.8, thresholds) == 2

    # Test with values outside the threshold range
    assert check_cat(0.9, thresholds) == 3
    assert check_cat(-0.1, thresholds) == 0

    # Test with empty thresholds
    assert check_cat(0.5, []) == 0

def test_category_free_nri_with_invalid_data():
    y_truth = np.array([0, "a", 1])
    y_ref = np.array([0.2, 0.3, 0.4])
    y_new = np.array([0.3, 0.4, 0.5])

    with pytest.raises(TypeError):
        category_free_nri(y_truth, y_ref, y_new)

def test_nri_valid_results():
    y_truth = np.array([0, 1, 0, 1, 0])
    y_ref = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.4])
    risk_thresholds = [0.3, 0.6]

    nri_events, nri_nonevents, total_nri = calculate_nri(y_truth, y_ref, y_new, risk_thresholds)
    
    assert isinstance(nri_events, float)
    assert isinstance(nri_nonevents, float)
    assert isinstance(total_nri, float)

def test_category_free_nri_valid_results():
    y_truth = np.array([0, 1, 0, 1, 0])
    y_ref = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.4])

    nri_events, nri_nonevents, total_nri = category_free_nri(y_truth, y_ref, y_new)

    assert isinstance(nri_events, float)
    assert isinstance(nri_nonevents, float)
    assert isinstance(total_nri, float)

def test_nri_pvalue_range():
    y_truth = np.array([0, 1, 0, 1, 0])
    y_ref = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.4])
    risk_thresholds = [0.3, 0.6]

    mean_nri, pval = nri_pvalue(y_truth, y_ref, y_new, risk_thresholds, num_bootstraps=50)
    assert isinstance(mean_nri, float)
    assert isinstance(pval, float)
    assert 0 <= pval <= 1

def test_nri_pvalue_identical_predictions():
    truth = np.array([0, 1, 0, 1])
    preds = np.array([0.2, 0.7, 0.1, 0.8])
    thresholds = [0.3, 0.6]

    mean_nri, pval = nri_pvalue(truth, preds, preds, thresholds, num_bootstraps=20)
    assert mean_nri == pytest.approx(0.0)
    assert pval == pytest.approx(1.0)

def test_nri_pvalue_significant_difference():
    truth = np.array([0, 0, 1, 1])
    ref = np.array([0.5, 0.5, 0.5, 0.5])
    new = np.array([0.2, 0.2, 0.8, 0.8])
    thresholds = [0.3, 0.6]

    mean_nri, pval = nri_pvalue(truth, ref, new, thresholds, num_bootstraps=20)
    assert mean_nri > 1.0
    assert pval < 0.01

def test_nri_pvalue_invalid_inputs():
    truth = np.array([0, 1])
    ref = np.array([0.1, 0.4])
    new = np.array([0.2])

    with pytest.raises(ValueError):
        nri_pvalue(truth, ref, new, [0.3, 0.6])

    with pytest.raises(TypeError):
        nri_pvalue([0, 1], [0.1, 0.4], [0.2, 0.3], [0.3, 0.6])

def test_nri_pvalue_reproducible():
    truth = np.array([0, 1, 0, 1])
    ref = np.array([0.1, 0.6, 0.2, 0.7])
    new = np.array([0.2, 0.7, 0.1, 0.8])
    thresholds = [0.3, 0.6]

    nri1, pval1 = nri_pvalue(truth, ref, new, thresholds, num_bootstraps=30)
    nri2, pval2 = nri_pvalue(truth, ref, new, thresholds, num_bootstraps=30)

    assert nri1 == pytest.approx(nri2)
    assert pval1 == pytest.approx(pval2)

