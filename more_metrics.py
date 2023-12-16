"""
File Name: more_metrics.py
Author: Lambert T Leong
Description: Contains code to compute area under the curve, net reclassification index, intergrated descrimination improvement index.
"""

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import matplotlib.colors as mcolors
import sys, random

def bootstrap_results(y_truth: np.ndarray, y_pred: np.ndarray, num_bootstraps: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform bootstrapping for ROC curve analysis.

    This function samples with replacement from the prediction indices to create bootstrap samples, and then computes the true positive rates (TPRs) and false positive rates (FPRs) for each bootstrap. It calculates and returns the mean TPRs and FPRs at a set of base thresholds, as well as the list of all TPRs and FPRs from each bootstrap iteration.

    Args:
        y_truth (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted probabilities.
        num_bootstraps (int, optional): Number of bootstrap iterations. Default is 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
            - base_thresh (np.ndarray): Base thresholds used for interpolation.
            - thresh_mean_tprs (np.ndarray): Mean true positive rates interpolated at base thresholds.
            - thresh_mean_fprs (np.ndarray): Mean false positive rates interpolated at base thresholds.
            - tprs (List[np.ndarray]): List of true positive rates for each bootstrap sample.
            - fprs (List[np.ndarray]): List of false positive rates for each bootstrap sample.
     Raises:
        ValueError: If all elements in y_truth are the same, or if y_truth or y_pred are empty.
        TypeError: If y_truth or y_pred are not numpy arrays.
    """
    # Check if inputs are numpy arrays
    if not isinstance(y_truth, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_truth and y_pred must be numpy arrays")

    # Check if inputs are non-empty
    if y_truth.size == 0 or y_pred.size == 0:
        raise ValueError("y_truth and y_pred must not be empty")
    
    # Check if all els are th same    
    if len(np.unique(y_truth)) < 2:
        raise ValueError("All elements in y_truth are the same. Need at least one positive and one negative sample for ROC AUC.")

    n_bootstraps = num_bootstraps 
    rng_seed = 42  # Control reproducibility
    rng = np.random.RandomState(rng_seed)
    thresh_tprs, thresh_fprs = [], []
    tprs, fprs = [],[]
    base_thresh = np.linspace(0, 1, 101)
    
    #for i in range(n_bootstraps):
    i=0
    while i<n_bootstraps:
        # Bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))

        if len(np.unique(y_truth[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        fpr, tpr, thresh = metrics.roc_curve(y_truth[indices], y_pred[indices])
        tprs.append(tpr)
        fprs.append(fpr)
        thresh = thresh[1:]
        thresh = np.append(thresh, [0.0])
        thresh = thresh[::-1]
        thresh_fpr = np.interp(base_thresh, thresh, fpr[::-1])
        thresh_tpr = np.interp(base_thresh, thresh, tpr[::-1])
        thresh_tprs.append(thresh_tpr)
        thresh_fprs.append(thresh_fpr)
        i+=1

    thresh_tprs = np.array(thresh_tprs)
    thresh_mean_tprs = thresh_tprs.mean(axis=0)
    thresh_fprs = np.array(thresh_fprs)
    thresh_mean_fprs = thresh_fprs.mean(axis=0)
    
    return base_thresh, thresh_mean_tprs, thresh_mean_fprs, tprs, fprs

def calculate_auc_with_ci(ground_truth: np.ndarray, predicted_probabilities: np.ndarray, num_bootstraps: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate the Area Under the Curve (AUC) with confidence intervals using bootstrapping.

    Args:
        ground_truth (np.ndarray): Ground truth labels.
        predicted_probabilities (np.ndarray): Predicted probabilities.
        num_bootstraps (int, optional): Number of bootstrap iterations. Default is 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        Base false positive rates, mean true positive rates, lower true positive rates,
        upper true positive rates, mean AUC, and AUC standard deviation.
    """

    base_thresh, _, _, tprs, fprs = bootstrap_results(ground_truth, predicted_probabilities, num_bootstraps)
    base_thresh=base_thresh
    bootstrapped_aucs,true_positive_rates = [], []
    for i, v in enumerate(tprs):
        fpr=fprs[i]
        auc = metrics.auc(fpr, v)
        bootstrapped_aucs.append(auc)
        tpr = np.interp(base_thresh, fpr, v)
        tpr[0] = 0.0
        true_positive_rates.append(tpr)
    true_positive_rates = np.array(true_positive_rates)
    mean_true_positive_rates = true_positive_rates.mean(axis=0)
    std_auc = np.std(bootstrapped_aucs)
    std_true_positive_rates = true_positive_rates.std(axis=0)
    mean_auc = metrics.auc(base_thresh, mean_true_positive_rates)
    std_auc = np.std(bootstrapped_aucs)
    upper_true_positive_rates = np.minimum(mean_true_positive_rates + std_true_positive_rates * 2, 1)
    lower_true_positive_rates = mean_true_positive_rates - std_true_positive_rates * 2
   
    return base_thresh, mean_true_positive_rates, lower_true_positive_rates, upper_true_positive_rates, mean_auc, std_auc

def plot_comparing_aucs(truth: np.ndarray, reference_model: np.ndarray, new_model: np.ndarray, n_bootstraps: int = 1000, save: bool = False):
    """
    Plots ROC curves and calculates AUC for reference and new models.

    Args:
        truth (np.ndarray): Ground truth labels.
        reference_model (np.ndarray): Predictions from the reference model.
        new_model (np.ndarray): Predictions from the new model.
        n_bootstraps (int, optional): Number of bootstraps for confidence intervals. Default is 1000.
        save (bool, optional): Whether to save the plot. Default is False.
    """

    y_truth = truth
    ref_model = reference_model
    new_model = new_model
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    ref_auc = metrics.auc(ref_fpr, ref_tpr)
    new_auc = metrics.auc(new_fpr, new_tpr)
    print('Reference AUC:', ref_auc)
    print('New AUC:', new_auc)

    base_fpr_ref, mean_tprs_ref, tprs_lower_ref, tprs_upper_ref, mean_auc_ref, std_auc_ref = calculate_auc_with_ci(y_truth, ref_model, n_bootstraps)
    base_fpr_new, mean_tprs_new, tprs_lower_new, tprs_upper_new, mean_auc_new, std_auc_new = calculate_auc_with_ci(y_truth, new_model, n_bootstraps)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(ref_fpr, ref_tpr, color='blue',
             lw=lw, label='Reference raw ROC (AUC = %0.2f)' % ref_auc, linestyle='--')
    plt.plot(base_fpr_ref, mean_tprs_ref, 'b', alpha=0.8, label=r'Reference mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_ref, (mean_auc_ref-2*std_auc_ref), (mean_auc_ref+2*std_auc_ref)),)
    plt.fill_between(base_fpr_ref, tprs_lower_ref, tprs_upper_ref, color='b', alpha=0.2)
    plt.plot(new_fpr, new_tpr, color='darkorange',
             lw=lw, label='New raw ROC (AUC = %0.2f)' % new_auc, linestyle='--')
    plt.plot(base_fpr_new, mean_tprs_new, 'darkorange', alpha=0.8, label=r'New mean ROC (AUC=%0.2f, CI=%0.2f-%0.2f)' % (mean_auc_new, (mean_auc_new-2*std_auc_new), (mean_auc_new+2*std_auc_new)),)
    plt.fill_between(base_fpr_new, tprs_lower_new, tprs_upper_new, color='darkorange', alpha=0.2)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity', fontsize=18)
    plt.ylabel('Sensitivity', fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')

def check_cat(prob: float, thresholds: List[float]) -> int:
    """
    Determine the category of a probability value based on given thresholds.

    Args:
    prob (float): The probability value to categorize.
    thresholds (List[float]): A list of threshold values defining the categories.

    Returns:
    int: The category index for the given probability.
    """
    for i, threshold in enumerate(thresholds):
        if prob <= threshold:
            return i
    return len(thresholds)

def make_cat_matrix(ref: np.ndarray, new: np.ndarray, indices: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """
    Create a matrix representing the transitions of categories from a reference to a new set.

    Args:
    ref (np.ndarray): Array of reference probabilities.
    new (np.ndarray): Array of new probabilities.
    indices (np.ndarray): Indices of elements to consider in the matrix.
    thresholds (List[float]): A list of threshold values defining the categories.

    Returns:
    np.ndarray: A matrix of category transitions.
    """
    num_cats = len(thresholds) + 1  # Adding one for the category above the highest threshold
    mat = np.zeros((num_cats, num_cats))
    for i in indices:
        row, col = check_cat(ref[i], thresholds), check_cat(new[i], thresholds)
        mat[row, col] += 1
    return mat

def nri(y_truth: np.ndarray, y_ref: np.ndarray, y_new: np.ndarray, risk_thresholds: List[float]) -> Tuple[float, float, float]:
    """
    Calculate the Net Reclassification Improvement (NRI) for a set of predictions.

    Args:
    y_truth (np.ndarray): The ground truth labels.
    y_ref (np.ndarray): The reference probabilities.
    y_new (np.ndarray): The new model probabilities.
    risk_thresholds (List[float]): A list of threshold values defining the risk categories.

    Returns:
    Tuple[float, float, float]: A tuple containing NRI for events, NRI for nonevents, and overall NRI.
    """
    event_index = np.where(y_truth == 1)[0]
    nonevent_index = np.where(y_truth == 0)[0]
    event_mat = make_cat_matrix(y_ref, y_new, event_index, risk_thresholds)
    nonevent_mat = make_cat_matrix(y_ref, y_new, nonevent_index, risk_thresholds)

    num_cats = len(risk_thresholds) + 1

    events_up = sum([event_mat[i, i+1:].sum() for i in range(num_cats - 1)])
    events_down = sum([event_mat[i, :i].sum() for i in range(1, num_cats)])
    nonevents_up = sum([nonevent_mat[i, i+1:].sum() for i in range(num_cats - 1)])
    nonevents_down = sum([nonevent_mat[i, :i].sum() for i in range(1, num_cats)])

    nri_events = (events_up / len(event_index)) - (events_down / len(event_index))
    nri_nonevents = (nonevents_down / len(nonevent_index)) - (nonevents_up / len(nonevent_index))

    return nri_events, nri_nonevents, nri_events + nri_nonevents

def track_movement(ref: np.ndarray, new: np.ndarray, indices: List[int]) -> Tuple[int, int]:
    """
    Track the movement (upward and downward) between two sets of predictions.

    Args:
        ref (np.ndarray): Reference predictions.
        new (np.ndarray): New predictions.
        indices (List[int]): List of data indices.

    Returns:
        Tuple[int, int]: Count of upward movements, count of downward movements.
    """
    up, down = 0, 0
    
    for i in indices:
        ref_val, new_val = ref[i], new[i]
        
        if ref_val < new_val:
            up += 1
        elif ref_val > new_val:
            down += 1
    
    return up, down

def category_free_nri(y_truth: np.ndarray, y_ref: np.ndarray, y_new: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate category-free Net Reclassification Improvement (NRI) for events and nonevents.

    Args:
        y_truth (np.ndarray): Ground truth labels.
        y_ref (np.ndarray): Reference predictions.
        y_new (np.ndarray): New predictions.

    Returns:
        Tuple[float, float, float]: NRI for events, NRI for nonevents, and total NRI.
    """
    if not np.issubdtype(y_truth.dtype, np.number):
        raise TypeError("All elements must be numerical")

    event_index = np.where(y_truth == 1)[0]
    nonevent_index = np.where(y_truth == 0)[0]
    events_up, events_down = track_movement(y_ref, y_new, event_index)
    nonevents_up, nonevents_down = track_movement(y_ref, y_new, nonevent_index)
    nri_events = (events_up / len(event_index)) - (events_down / len(event_index))
    nri_nonevents = (nonevents_down / len(nonevent_index)) - (nonevents_up / len(nonevent_index))
    
    return nri_events, nri_nonevents, nri_events + nri_nonevents

def area_between_curves(y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the area between two curves.

    Args:
        y1 (np.ndarray): First curve values.
        y2 (np.ndarray): Second curve values.

    Returns:
        Tuple[float, float, float]: Positive area, negative area, and net area between the curves.
    """
    diff = y1 - y2
    posPart = np.maximum(diff, 0)
    negPart = -np.minimum(diff, 0)
    posArea = np.trapz(posPart)
    negArea = np.trapz(negPart)
    netArea = posArea - negArea
    return posArea, negArea, netArea

def plot_idi(y_truth: np.ndarray, ref_model: np.ndarray, new_model: np.ndarray, thresholds: dict, save: bool = False):
    """
    Plot the Integrated Discrimination Improvement (IDI) curve.

    Args:
        y_truth (np.ndarray): Ground truth labels.
        ref_model (np.ndarray): Reference model predictions.
        new_model (np.ndarray): New model predictions.
        thresholds (dict): Dictionary of thresholds with their corresponding labels.
        save (bool, optional): Whether to save the plot. Default is False.
    """
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    base, mean_tprs, mean_fprs,_,_ = bootstrap_results(y_truth, new_model, 100)
    base2, mean_tprs2, mean_fprs2,_,_ = bootstrap_results(y_truth, ref_model, 100)
    is_pos, is_neg, idi_event = area_between_curves(mean_tprs, mean_tprs2)
    ip_pos, ip_neg, idi_nonevent = area_between_curves(mean_fprs2, mean_fprs)
    
    print('IS positive', round(is_pos, 2), 'IS negative', round(is_neg, 2), 'IDI events', round(idi_event, 2))
    print('IP positive', round(ip_pos, 2), 'IP negative', round(ip_neg, 2), 'IDI nonevents', round(idi_nonevent, 2))
    print('IDI =', round(idi_event + idi_nonevent, 2))
    
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    lw = 2
    plt.plot(base, mean_tprs, 'black', alpha=0.5, label='Events New (New)')
    plt.plot(base, mean_fprs, 'red', alpha=0.5, label='Nonevents New (New)')
    plt.plot(base2, mean_tprs2, 'black', alpha=0.7, linestyle='--', label='Events Reference (Ref)')
    plt.plot(base2, mean_fprs2, 'red', alpha=0.7, linestyle='--', label='Nonevents Reference (Ref)')
    plt.fill_between(base, mean_tprs, mean_tprs2, color='black', alpha=0.1,
                     label='Integrated Sensitivity (area = %0.2f)' % idi_event)
    plt.fill_between(base, mean_fprs, mean_fprs2, color='red', alpha=0.1,
                     label='Integrated Specificity (area = %0.2f)' % idi_nonevent)

    def nri_annotation(plt, threshold, label, color):
        x_pos = base[threshold]
        x_offset = 0.02
        x_offset2 = x_offset
        text_y_offset = 0.01
        text_y_offset2 = text_y_offset
        if threshold == 2:
            text_y_offset = 0.04
            text_y_offset2 = 0.04
            x_offset2 = 0.05
            print(x_pos + x_offset, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]]) + text_y_offset),
                  x_pos, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]])))
        text_y_events = np.mean([mean_tprs2[threshold], mean_tprs[threshold]]) + text_y_offset
        text_y_nonevents = np.mean([mean_fprs[threshold], mean_fprs2[threshold]]) + text_y_offset2

        plt.axvline(x=threshold/100, color=color, linestyle='--', alpha=0.5, label=label)
        plt.annotate('', xy=(x_pos + 0.02, mean_tprs2[threshold + 1]), xycoords='data',
                     xytext=(x_pos + 0.02, mean_tprs[threshold]), textcoords='data', arrowprops={'arrowstyle': '|-|'})
        plt.annotate('NRI$_{events}$ = %0.2f' % (mean_tprs[threshold] - mean_tprs2[threshold]),
                     xy=(x_pos + x_offset, text_y_events), xycoords='data',
                     xytext=(x_pos + x_offset, text_y_events),
                     textcoords='offset points', fontsize=15)
        plt.annotate('', xy=(x_pos + 0.02, mean_fprs[threshold]), xycoords='data',
                     xytext=(x_pos + 0.02, mean_fprs2[threshold]), textcoords='data',
                     arrowprops=dict(arrowstyle='|-|', color='r'))
        plt.annotate('NRI$_{nonevents}$ = %0.2f' % (mean_fprs2[threshold] - mean_fprs[threshold]),
                     xy=(x_pos + x_offset2, text_y_nonevents), xycoords='data',
                     xytext=(x_pos + x_offset2, text_y_nonevents),
                     textcoords='offset points', fontsize=15)
        print('Threshold =', round(x_pos, 2), 'NRI events =', round(mean_tprs[threshold] - mean_tprs2[threshold], 4),
              'NRI nonevents =', round(mean_fprs2[threshold] - mean_fprs[threshold], 4), 'Total =',
              round((mean_tprs[threshold] - mean_tprs2[threshold]) + (mean_fprs2[threshold] - mean_fprs[threshold]), 4))

    def generate_distinct_colors(list_size: int, avoid_red: bool = True, avoid_dark: bool = True) -> list:
        """
        Generate a list of distinct colors, optionally avoiding red and dark (near black) colors.
    
        Args:
            list_size (int): The number of distinct colors to generate.
            avoid_red (bool): Whether to avoid red colors. Defaults to True.
            avoid_dark (bool): Whether to avoid dark colors. Defaults to True.
    
        Returns:
            list: A list of colors in hexadecimal format.
        """
        colors = []
        # Define the HSV range to avoid red and dark colors
        hue_start = 0.1 if avoid_red else 0
        hue_end = 0.9 if avoid_red else 1
        saturation = 0.7
        value = 0.9 if avoid_dark else 0.5
        # Use HSV tuples to evenly space out colors and convert to RGB
        for i in np.linspace(hue_start, hue_end, list_size, endpoint=False):
            rgb_color = mcolors.hsv_to_rgb((i, saturation, value))  # Saturation and Value adjusted for color brightness
            hex_color = mcolors.rgb2hex(rgb_color)
            colors.append(hex_color)
        
        return colors
        
    colors = generate_distinct_colors(len(thresholds))    
    for c, i in enumerate(thresholds):
        nri_annotation(plt, int(i), thresholds[i], colors[c])
        
    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Calculated Risk', fontsize=18)
    plt.ylabel('Sensitivity (black), 1 - Specificity (red)', fontsize=18)
    plt.legend(loc="upper right", fontsize=11)
    plt.legend(loc=0, fontsize=11, bbox_to_anchor=(0, 0, 1.2, .9))
    plt.gca().set_aspect('equal', adjustable='box')

    if save:
        plt.savefig('idi_curve.png', dpi=300, bbox_inches='tight')
    
    plt.show()
