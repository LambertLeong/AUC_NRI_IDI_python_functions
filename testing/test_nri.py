import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import nri, category_free_nri

'''
def test_nri_normal_case():
    y_truth = np.array([0, 1, 1, 0, 1])
    y_ref = np.array([0.2, 0.4, 0.6, 0.5, 0.7])
    y_new = np.array([0.3, 0.5, 0.7, 0.4, 0.8])
    risk_thresholds = [0.3, 0.6]

    nri_events, nri_nonevents, total_nri = nri(y_truth, y_ref, y_new, risk_thresholds)
    
    # Assertions based on expected calculations
    # ...

def test_category_free_nri_normal_case():
    y_truth = np.array([0, 1, 1, 0, 1])
    y_ref = np.array([0.2, 0.4, 0.6, 0.5, 0.7])
    y_new = np.array([0.3, 0.5, 0.7, 0.4, 0.8])

    nri_events, nri_nonevents, total_nri = category_free_nri(y_truth, y_ref, y_new)
    
    # Assertions based on expected calculations
    # ...

def test_nri_with_empty_arrays():
    with pytest.raises(ValueError):
        nri(np.array([]), np.array([]), np.array([]), [0.5])

def test_nri_with_single_class():
    y_truth = np.array([1, 1, 1])
    y_ref = np.array([0.7, 0.8, 0.9])
    y_new = np.array([0.8, 0.9, 1.0])
    risk_thresholds = [0.5]

    nri_events, nri_nonevents, total_nri = nri(y_truth, y_ref, y_new, risk_thresholds)
    # Assertions for single-class scenario

'''
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

    nri_events, nri_nonevents, total_nri = nri(y_truth, y_ref, y_new, risk_thresholds)
    
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

