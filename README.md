# AUC_NRI_IDI_python_functions
[![DOI](https://zenodo.org/badge/365029676.svg)](https://zenodo.org/doi/10.5281/zenodo.4741234)

Custom python functions to help you further analyse machine learning models and diagnostic test.

Will help you make plots and compute evaluation metrics as seen in [Nature Article, Leong et al. 2021](https://www.nature.com/articles/s43856-021-00024-0?source=post_page-----f87a8ec6937b--------------------------------#Fig3)

![From Leong et. al. 2021](https://github.com/LambertLeong/AUC_NRI_IDI_python_functions/blob/main/idi_auc.png)

Metrics to compute and plot:
* AUC = Area Under the Curve
* NRI = Net Reclassification Index
* IDI = Integrated Discrimination Improvement
* Functions to compute bootstrap p-values for AUC and NRI differences

Run "example.ipynb" Jupyter notebook to see and use functions

## Installation
```bash
pip install -r requirements.txt
pip install .
```

## Running tests
```bash
pytest -q
```

## Formulas
**AUC**
\[AUC = \int_0^1 TPR(FPR)\, dFPR \]

**NRI**
\[NRI = (P_{\text{up}|\text{event}} - P_{\text{down}|\text{event}}) + (P_{\text{down}|\text{non-event}} - P_{\text{up}|\text{non-event}})\]

**IDI**
\[IDI = (\bar{p}_{\text{new},1} - \bar{p}_{\text{ref},1}) - (\bar{p}_{\text{new},0} - \bar{p}_{\text{ref},0})\]

## Usage
Import the package and call any metric helpers. See the example notebook for a detailed walkthrough.

Code and concepts further explained in the following post: "[Area Under the Curve and Beyond](https://www.lambertleong.com/thoughts/AUC-IDI-NRI)" or "[On Medium/Towards Data Science](https://medium.com/towards-data-science/area-under-the-curve-and-beyond-f87a8ec6937b)"

---
contact [Lambert Leong](https://www.lambertleong.com)
