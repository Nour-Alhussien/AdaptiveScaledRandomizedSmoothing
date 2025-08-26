# AdaptiveScaledRandomizedSmoothing
This project trains a tabular neural network on a balanced UNSW-NB15 split, generates adversarial examples with the Adversarial Robustness Toolbox (ART), and evaluates two certification schemes:

* Original randomized smoothing (baseline)

* AdaptiveSmoothEntropy — an adaptive variant that: injects entropy-aware noise during training and uses a margin–confidence proxy to adapt the number of samples during certification.

The code is written as a Colab-friendly notebook, but also runs locally with a standard Python environment.

# Training: 
Simple 4-layer MLP for tabular data (TabularNN) and Adam/AdamW optimizers, optional StepLR scheduler

# Attacks (ART): 
* FGSM, PGD, DeepFool, HopSkipJump (HSJ), Carlini–Wagner (CW-L2), ZOO, (JSMA optional)

* Tabular attack: LowProFool: weighted-norm, low-profile perturbations using feature importance

# Certification: 
* Baseline Randomized Smoothing (Smooth)
* AdaptiveSmoothEntropy (entropy-aware training + margin-based adaptive sampling)

# Evaluation: 
* Clean accuracy
*  Adversarial accuracy (per attack)
*  Certified accuracy (fraction of points whose certified class matches the true label)
*  5-fold evaluation over the test set indices for adversarial files
*  Timing per attack.

# Environment & Setup (Python dependencies):
pandas, numpy, scikit-learn, torch, torchvision (GPU optional), adversarial-robustness-toolbox (ART), scipy, statsmodels, matplotlib

--- pip install torch pandas numpy scikit-learn adversarial-robustness-toolbox scipy statsmodels matplotlib

# The main steps to run the adaptive randomized smoothing method:

* Load & prepare data:

    - Reads the balanced UNSW train/test CSVs

    - Splits features/labels and converts to PyTorch tensors

    - Detects categorical columns by the cat__ prefix (+ some extra indicators)

* Train a base model

    - TabularNN (MLP) trained with cross-entropy

    - Reports epoch losses to evaluation_results.txt

    - Build an ART classifier

    - PyTorchClassifier wrapper with clip_values=(0,1) (change if your features aren’t in [0,1])

* Generate adversarial examples

    - Attacks: FGSM, PGD, DeepFool, HSJ, CW-L2, ZOO, (JSMA optional)

    - LowProFool (custom): uses Pearson correlation-based feature importance for weighted perturbations; includes bounds and a small step size to be less aggressive

    - Saves raw adversarial arrays (e.g., X_adv_pgd_unsw_M.txt)

    - Restore categorical features

    - After attack, categorical indices are copied back from clean test data to maintain domain validity

* Evaluate

 - Clean accuracy on test set for each attack
 - Adversarial accuracy
 - Evaluation time

# The code for Adaptive model (AdaptiveSmoothEntropy) find:

   - Entropy-aware noise mixing during later training epochs

   - Certification via margin–confidence proxy and adaptive sampling (n scaled by margin & confidence)

   - Reports accuracy and certified accuracy
# The code for Original smoothing (Smooth) find: 

- Standard two-stage sampling (n0, n)
-  Clopper–Pearson LCB

5-fold evaluation over test indices for adversarial files; averages are reported
