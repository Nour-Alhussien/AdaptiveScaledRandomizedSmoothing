# AdaptiveScaledRandomizedSmoothing

This project trains a tabular neural network on a balanced UNSW-NB15 split, generates adversarial examples with the Adversarial Robustness Toolbox (ART), and evaluates two certification schemes:

1. Original randomized smoothing (baseline)

2. AdaptiveSmoothEntropy — an adaptive variant that: injects entropy-aware noise during training and uses a margin–confidence proxy to adapt the number of samples during certification.

The code is written as a Colab-friendly notebook, but also runs locally with a standard Python environment.

* Training: Simple 4-layer MLP for tabular data (TabularNN) and Adam/AdamW optimizers, optional StepLR scheduler

* Attacks (ART): FGSM, PGD, DeepFool, HopSkipJump (HSJ), Carlini–Wagner (CW-L2), ZOO, (JSMA optional)

* Tabular attack: LowProFool: weighted-norm, low-profile perturbations using feature importance

* Certification: Baseline Randomized Smoothing (Smooth) and AdaptiveSmoothEntropy (entropy-aware training + margin-based adaptive sampling)

* Evaluation: Clean accuracy, Adversarial accuracy (per attack), Certified accuracy (fraction of points whose certified class matches the true label), 5-fold 
              evaluation over the test set indices for adversarial files, and Timing per attack.

* Environment & Setup (Python dependencies): pandas, numpy, scikit-learn, torch, torchvision (GPU optional), adversarial-robustness-toolbox (ART), scipy, 
                                             statsmodels, matplotlib
  
