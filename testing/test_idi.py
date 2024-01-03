import numpy as np
import pytest, sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from more_metrics import plot_idi

def test_plot_idi_output_format():
    # Sample input data
    y_truth = np.array([0, 1, 1, 0, 1])
    ref_model = np.array([0.1, 0.4, 0.35, 0.6, 0.7])
    new_model = np.array([0.2, 0.5, 0.3, 0.7, 0.8])
    thresholds = {50: 'mid', 70: 'high', 90: 'very high'}

    # Call the function
    output = plot_idi(y_truth, ref_model, new_model, thresholds, num_bootstraps=100, show=False, save=False)

    # Check that the output is a dictionary
    assert isinstance(output, dict), "Output should be a dictionary"

    # Check for the keys in the output
    assert 'plot' in output, "Output should have a 'plot' key"
    assert 'IDI' in output, "Output should have an 'IDI' key"
    assert 'NRI' in output, "Output should have an 'NRI' key"

    # Check the types of the values
    assert isinstance(output['IDI'], dict), "'IDI' should be a dictionary"
    assert isinstance(output['NRI'], dict), "'NRI' should be a dictionary"

    # Check the structure of the 'IDI' dictionary
    assert all(key in output['IDI'] for key in ['IDI', 'IDI Events', 'IDI Nonevents', 'IS Positive', 'IS Negative', 'IP Positive', 'IP Negative']), "IDI dictionary has missing keys"

    # Check the structure of the 'NRI' dictionary
    #assert all(isinstance(output['NRI'][thresh], list) for thresh in thresholds), "NRI dictionary should contain lists for each threshold"


def test_plot_idi_input_validation():
    # Test with invalid input types
    with pytest.raises(TypeError):
        plot_idi('invalid', 'invalid', 'invalid', {}, num_bootstraps=100, show=False, save=False)
    
    # Test with mismatched array lengths
    with pytest.raises(ValueError):
        plot_idi(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4]), {50: 'mid'}, num_bootstraps=100, show=False, save=False)

    # Test with invalid threshold dictionary
    with pytest.raises(ValueError):
        plot_idi(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.6, 0.2]), np.array([0.3, 0.5, 0.7, 0.3]), {'invalid': 'mid'}, num_bootstraps=100, show=False, save=False)

    # Test with invalid num_bootstraps type
    with pytest.raises(TypeError):
        plot_idi(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.6, 0.2]), np.array([0.3, 0.5, 0.7, 0.3]), {50: 'mid'}, num_bootstraps='100', show=False, save=False)

