# Python-Portfolio
My code portfolio showcasing several advanced statistical and machine learning techniques applied to scientific and simulated data.

---

### `Best_Fit_Line.py`

This script constructs a best-fit line for data points with arbitrary two-dimensional uncertainties. It leverages Bayesian techniques and Gaussian mixture models to account for measurement errors and identify statistical outliers. The best-fit line is computed using maximum likelihood estimation, and the posterior probability of each data point being an outlier is visualised through colour mapping. The final output includes sample posterior draws and plots that communicate the reliability and robustness of the fitting process.

**To run:**
```bash
pip install numpy matplotlib scipy emcee corner
```

```python
import Best_Fit_Line
```

---

### `Expectation_maximisation.py`

This script generates a simulated dataset of clustered points and applies the Expectation-Maximisation (EM) algorithm to estimate the parameters of the underlying distributions. The log-likelihood and its change over iterations are plotted, helping users visualise the convergence process. The clustering results are shown at each step, along with uncertainty ellipses representing three standard deviations. This is useful for understanding unsupervised learning and model fitting.

**To run:**
```bash
pip install pandas numpy matplotlib scikit-learn
```

```python
import Expectation_maximisation
```

---

### `Gaussian_model_prediction.py`

This script models the brightness of a star using observational time-series data spanning three months. It uses a Lomb-Scargle periodogram to identify significant periodicities and fits a Gaussian Process model with different kernels (Matern 3/2, Exponential Sine Squared, and optionally Rational Quadratic) to predict future brightness over the next two months. The output includes confidence intervals and posterior draws obtained via MCMC sampling using `emcee`, showcasing uncertainty in long-term astrophysical forecasting.

>  **Note:** This script may take 10–30 minutes to execute due to the MCMC sampling step.

**To run:**
```bash
pip install numpy matplotlib astropy george scipy emcee
```

```python
import Gaussian_model_prediction
```

---

### `Hamiltonian_monte_carlo_sampler.py`

This code simulates Hamiltonian Monte Carlo (HMC) sampling in a two-dimensional square domain. It includes a custom leapfrog integrator to simulate trajectories of particles under a defined potential energy surface. The sampler preserves energy well and is used to draw samples from a target distribution. The script plots particle paths, kinetic/potential/total energy, sampling chains, and posterior distributions using corner plots. This provides an intuitive look at the power of HMC for high-efficiency sampling.

**To run:**
```bash
pip install numpy matplotlib scipy chainconsumer pandas corner tqdm
```

```python
import Hamiltonian_monte_carlo_sampler
```

---

### `principal_component_analysis.py`

This script applies Principal Component Analysis (PCA) to facial image data from the Labeled Faces in the Wild (LFW) dataset. It reduces the data's dimensionality and visualises the most significant principal components, or “eigenfaces.” The script also reconstructs the original images using a subset of principal components, allowing visual evaluation of dimensionality reduction. The cumulative explained variance plot helps determine the number of components required to retain key information in the dataset.

**Key features:**
- Loads and processes grayscale facial images
- Applies PCA with whitening and randomised SVD
- Visualises top 50 eigenfaces
- Reconstructs faces and compares to originals
- Plots explained variance vs number of components

**To run:**
```bash
pip install numpy matplotlib scikit-learn
```

```python
import principal_component_analysis
```

---

### `Stan_model_prediction.py`

This script models radioactive decay using Bayesian inference with Stan and CmdStanPy. It analyses real-world measurement data to infer the decay rate and potential measurement biases introduced by different detectors.

#### Part 1:
- Estimates the decay constant α assuming unknown initial masses and random production times.
- Uses `decay.stan` to fit the Bayesian model to real data (with uncertainty).
- Plots the posterior of α and initial masses for individual samples.

#### Part 2:
- Extends the model to account for biases in three measurement detectors.
- Uses `decay2.stan` to model measurement errors.
- Outputs posteriors for detector biases and decay rate.

**Key features:**
- Handles noisy real-world data with uncertainties.
- Visualises posterior distributions and sampler diagnostics.
- Implements full Bayesian workflow using Stan models.

**To run:**
- Ensure [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) is installed and configured.
- Place `decay.stan` and `decay2.stan` in the working directory.
```bash
pip install numpy pandas matplotlib seaborn cmdstanpy
```

```python
import Stan_model_prediction
```

> **Note:** Time is represented in seconds internally but converted to days in plots. All Stan data is passed as reproducible Python dictionaries.

---
