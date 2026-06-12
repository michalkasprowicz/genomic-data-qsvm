# Analysis of genomic data for cancer mutations using quantum support vector machine algorithm (QSVM).

## Project Overview

This project implements a **comprehensive comparison of quantum and classical SVM algorithms** in the context of brain tumor classification (GBM and LGG) based on TCGA genetic data. The study includes robustness analysis of different quantum feature maps against various types and levels of noise in medical data.

### Main Research Objectives

- **Performance comparison** of quantum and classical SVM algorithms
- **Robustness analysis** of different quantum feature maps against data noise
- **Practicality assessment** of quantum algorithms in real-world medical applications

### Experiment Scope

- **11 datasets**: 1 clean + 10 noisy (additive and substitutional, 1-20%)
- **4 experiment types**: ZZ, Pauli, Z, Amplitude
- **6 quantum feature maps**: ZZ1, ZZ2, Pauli1, Pauli2, Z1, Z2
- **3 C parameters**: 0.1, 1.0, 10.0
- **10-fold cross-validation** for each combination
- **Classical SVM** as baseline

## Project Structure

```
├── 📁 Data
│   ├── qsvm.py                # Main experiment controller
│   ├── qsvm1_zz.py            # Experiment 1: ZZ Feature Maps
│   ├── qsvm2_pauli.py         # Experiment 2: Pauli Feature Maps
│   ├── qsvm3_z.py             # Experiment 3: Z Feature Maps
│   ├── qsvm4_amplitude.py     # Experiment 4: Amplitude Encoding
│   └── dane/                  # TCGA datasets
│       ├── TCGA_GBM_LGG_Mutations_all.csv   # high dimensional data
│       ├── TCGA_GBM_LGG_Mutations_clean.csv # low dimensional data
│       ├── zaszumione/                      # Substitutional noise
│       └── zaszumione_rozszerzone/          # Additive noise
│
├── 📁 Results
│
├── 📁 Configuration
│   ├── environment.yml       # Conda environment
│   └── requirements.txt      # Python dependencies
│
└── 📁 Side experiments
    ├── experiments.py        # Experiments file 
    └── run_experiment.sh     # Shell script to run cloud computing using multi thread option
```

## Side Experiments Overview

The directory contains **side experiments** that extend the main quantum brain tumor classification project. The experiments focus on **analyzing the impact of genetic data complexity** and **different gene subsets** on the effectiveness of quantum SVM algorithms.

### Main Research Objectives

- **Genetic complexity analysis**: Impact of mutation count on classification performance
- **Gene subset testing**: Comparison of different gene groups (frequently/moderately/rarely mutated)
- **Multi-core optimization**: Utilizing VAST.AI cloud computing
- **Feature map comparison**: Testing different quantum feature maps on diversified data

### Experiment Scope

- **2 main experiments**: Complexity analysis + Gene subsets
- **4 gene subsets**: All, frequently mutated, moderately mutated, rarely mutated
- **14 quantum feature maps**: Pauli, Z, Amplitude with different parameters
- **3 complexity levels**: Low, medium, high (based on quartiles)
- **Multi-core processing**: Optimization for cloud computing

## Experimental Methodology

### Experiment: Genetic Complexity Analysis

#### Complexity Definition
Genetic data complexity is defined based on **the number of genetic mutations per case**:

```python
# Calculate mutation count for each case
mutation_counts = X.sum(axis=1)

# Classification into complexity levels based on quartiles
low_threshold = mutation_counts.quantile(0.25)    # 25% quartile
high_threshold = mutation_counts.quantile(0.75)   # 75% quartile
```

#### Complexity Levels

1. **Low Complexity**
   - **Criterion**: `mutation_counts ≤ 25% quartile`
   - **Characteristics**: Cases with few mutations
   - **Expectations**: Better performance of linear algorithms

2. **Medium Complexity**
   - **Criterion**: `25% quartile < mutation_counts < 75% quartile`
   - **Characteristics**: Cases with moderate mutation count
   - **Expectations**: Greatest advantage of quantum algorithms

3. **High Complexity**
   - **Criterion**: `mutation_counts ≥ 75% quartile`
   - **Characteristics**: Cases with many mutations
   - **Expectations**: Better performance of nonlinear algorithms

### Parameter Configuration

You can customize parameters in the `experiments.py` file:

```python
# Experiment selection
RUN_GENE_SUBSETS_EXPERIMENT = True    # Gene subsets experiment
RUN_COMPLEXITY_EXPERIMENT = False     # Complexity experiment
RUN_FEATURE_MAPPINGS_EXPERIMENT = False  # Feature mappings experiment

# Quantum parameters
QUANTUM_SHOTS = 50                    # Number of shots (reduced for performance)
QUANTUM_TIMEOUT = 300                 # 5-minute timeout
MAX_FEATURE_DIMENSION = 8             # Maximum feature dimension

# Multi-core parameters
USE_MULTIPROCESSING = True            # Enable parallel processing
MAX_WORKERS = None                    # Automatic core detection
```

## Installation and Configuration

### System Requirements

- **Python**: 3.9
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Multi-core processor (for parallel processing)
- **Disk**: ~5GB free space

### Method 1: Conda (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd kod_sierpien

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate MK_QSVM
```

### Method 2: Manual Installation

```bash
# Create environment
conda create -n MK_QSVM python=3.9
conda activate MK_QSVM

# Install basic libraries
conda install -c conda-forge numpy=1.24.3 pandas=2.0.3 scikit-learn=1.3.0
conda install -c conda-forge matplotlib=3.7.2 seaborn=0.12.2 jupyter=1.0.0

# Install quantum libraries
pip install qiskit==0.44.1 qiskit-aer==0.12.2 qiskit-machine-learning==0.6.0
pip install dimod==0.12.8 umap-learn==0.5.3 plotly==5.16.1
```

### Method 3: Requirements.txt

```bash
# Create virtual environment
python -m venv qsvm-env
source qsvm-env/bin/activate  # Linux/Mac
# or
qsvm-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Basic Launch

```bash
# Activate environment
conda activate MK_QSVM

# Run all experiments
python qsvm.py
```

### Running Individual Experiments

```bash
# Experiment 1: ZZ Feature Maps
python qsvm1_zz.py

# Experiment 2: Pauli Feature Maps
python qsvm2_pauli.py

# Experiment 3: Z Feature Maps
python qsvm3_z.py

# Experiment 4: Amplitude Encoding
python qsvm4_amplitude.py
```

### Parameter Configuration

You can customize parameters in the `qsvm.py` file:

```python
# Data parameters
DATA_FILES = [
    'dane/TCGA_GBM_LGG_Mutations_all.csv',
    # Add or remove files as needed
]

# Experiment parameters
RUN_CLASSIC_SVM = True      # Classical SVM
RUN_QUANTUM_SVM = True      # Quantum SVM
RUN_HYBRID_APPROACH = True  # Hybrid approach

# Dimensionality reduction parameters
USE_PCA = True
PCA_COMPONENTS = 12
```

## Experimental Methodology

### 1. Data Preparation

- **Source**: TCGA (The Cancer Genome Atlas) - GBM/LGG data
- **Target variable**: `Primary_Diagnosis`
- **Features**: Genetic mutations, demographic data, clinical features
- **Processing**: Standardization, dimensionality reduction (PCA), train/test split

### 2. Quantum Feature Maps

#### ZZFeatureMap
- **Structure**: Hadamard gates + Z rotations + ZZ entanglements
- **Properties**: Local encoding with quantum correlations
- **Implementation**: `ZZFeatureMap` from Qiskit

#### PauliFeatureMap
- **Structure**: Utilizes all Pauli axes (X, Y, Z)
- **Properties**: Richer encoding with stronger entanglements
- **Implementation**: `PauliFeatureMap` from Qiskit

#### ZFeatureMap
- **Structure**: Only Hadamard gates and Z rotations
- **Properties**: Simpler, more stable
- **Implementation**: `ZFeatureMap` from Qiskit

#### Amplitude Encoding
- **Structure**: Amplitude encoding with different normalizations
- **Properties**: Custom kernel `K(x,y) = (x·y)²`
- **Implementation**: Custom `AmplitudeKernel` class

### 4. Validation and Metrics

- **Cross-validation**: 10-fold for QSVM, 5-fold for classical SVM
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Comparison**: Classical SVM vs Quantum SVM

### Key Results

Analysis of **81 experiments** shows:

1. **Classical SVM**: 100% accuracy in all experiments
2. **AMPLITUDE**: 96.7% accuracy
3. **ZZ**: 86.1% accuracy
4. **PAULI**: 81.4% accuracy
5. **Z**: 66.0% accuracy

## Troubleshooting

### Library Errors

```bash
# If version conflicts occur
conda clean --all
conda env remove -n MK_QSVM
conda env create -f environment.yml
```

### Memory Issues

1. Reduce data size in `DATA_FILES`
2. Disable some feature maps
3. Reduce number of PCA components

### Installation Check

```bash
# Check local simulator
python -c "from qiskit import Aer; print('Local simulator OK')"

# Check quantum libraries
python -c "from qiskit_machine_learning import QSVC; print('QSVC OK')"
```

## Academic Context

### Research Area

- **Quantum Machine Learning**
- **Medical Classification**
- **Noise Robustness**
- **Quantum Optimization**

### Key files to check:

- `environment.yml` - environment configuration
- `qsvm.py` - experiment parameters
- Cache files - experiment progress
- Result files - detailed logs

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The project is intended for **research and educational purposes**. All data comes from publicly available TCGA sources.

## Acknowledgments

- **TCGA** for providing genetic data
- **IBM Qiskit** for quantum framework
- **VAST.AI** for cloud platform
- **Adam Mickiewicz University** for research support

---

**Last update:** 2025-01-09
**Version:** 1.0
**Status:** Ready for experiment execution