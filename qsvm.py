import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import json
import gc
import signal
import multiprocessing
from multiprocessing import Pool

# Import bibliotek kwantowych
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import dimod

# Dodanie zalecanego zamiennika dla ZZFeatureMap
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.data_preparation import ZFeatureMap

# Dodanie bibliotek do kodowania amplitudowego
from qiskit.circuit import QuantumCircuit
from qiskit.extensions import Initialize
import scipy.linalg as la

# Dodanie biblioteki UMAP
import umap

# Maksymalny czas trwania eksperymentu 
MAX_EXECUTION_TIME = 24 * 60 * 60 * 7 * 3 

def timeout_handler(signum, frame):
    print("\n\n======= PRZEKROCZONO MAKSYMALNY CZAS WYKONANIA =======")
    print(f"Eksperyment został przerwany po {MAX_EXECUTION_TIME/3600:.1f} godzinach.")
    sys.exit(1)

# Ustawienie obsługi sygnału
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(MAX_EXECUTION_TIME)

# ----------------- CZĘŚĆ 0: PARAMETRY KONFIGURACYJNE -----------------

# Parametry danych
DATA_FILES = [
    'dane/TCGA_GBM_LGG_Mutations_all.csv',
    'dane/zaszumione_rozszerzone/TCGA_GBM_LGG_Mutations_noise_1percent_added.csv',
    'dane/zaszumione_rozszerzone/TCGA_GBM_LGG_Mutations_noise_5percent_added.csv',
    'dane/zaszumione_rozszerzone/TCGA_GBM_LGG_Mutations_noise_10percent_added.csv',
    'dane/zaszumione_rozszerzone/TCGA_GBM_LGG_Mutations_noise_15percent_added.csv',
    'dane/zaszumione_rozszerzone/TCGA_GBM_LGG_Mutations_noise_20percent_added.csv',
    'dane/zaszumione/TCGA_GBM_LGG_Mutations_noise_1percent_substituted.csv',
    'dane/zaszumione/TCGA_GBM_LGG_Mutations_noise_5percent_substituted.csv',
    'dane/zaszumione/TCGA_GBM_LGG_Mutations_noise_10percent_substituted.csv',
    'dane/zaszumione/TCGA_GBM_LGG_Mutations_noise_15percent_substituted.csv',
    'dane/zaszumione/TCGA_GBM_LGG_Mutations_noise_20percent_substituted.csv'
]
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Parametry redukcji wymiarowości
USE_PCA = True
USE_TSNE = False
USE_UMAP = False
PCA_COMPONENTS = 14
TSNE_COMPONENTS = 3
TSNE_PERPLEXITY = 100
TSNE_LEARNING_RATE = 50
TSNE_MAX_ITER = 1000
UMAP_COMPONENTS = 14
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.8
UMAP_METRIC = 'euclidean'
EVALUATE_SILHOUETTE = False
OPTIMAL_SILHOUETTE_SCORE = 0.1964

# Wybór eksperymentów do przeprowadzenia
RUN_CLASSIC_SVM = True
RUN_QUANTUM_SVM = True
RUN_HYBRID_APPROACH = True

# Parametry klasycznego SVM
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['linear', 'rbf', 'poly']
}
SVM_CV = 5

# Parametry kwantowego SVM
BACKEND_NAME = 'qasm_simulator'
C_VALUES = [0.1, 1.0, 10.0]
QSVM_CV = 10

# Parametry wyżarzania kwantowego
NUM_READS = 100
QUBO_PENALTY = 10.0

# Parametry analizy cech
IMPORTANCE_THRESHOLD = 0.01

# Parametry wyjściowe
OUTPUT_DIR = f'wyniki/2025-08-04-dim_reduction-{QSVM_CV}-fold'

# Parametry IBM Quantum Cloud
USE_IBM_QUANTUM = False
IBM_BACKEND = 'qasm_simulator'
IBM_REAL_BACKEND = 'qasm_simulator'
IBM_TOKEN = None
IBM_INSTANCE = None
IBM_MAX_SHOTS = 1024
IBM_OPTIMIZATION_LEVEL = 1
IBM_RESILIENCE_LEVEL = 1

# Upewnij się, że katalog wyjściowy istnieje
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Przekierowanie wyjścia do pliku i konsoli
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Funkcja do zapisywania szczegółowych metryk
def save_metrics(y_true, y_pred, model_name):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = "N/A"
    
    print(f"\nSzczegółowe metryki dla modelu {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nMacierz pomyłek:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

# Funkcja do zapisywania wyników pośrednich
def save_results_cache(results_dict, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(results_dict, f)

# Funkcja do wczytywania wyników pośrednich
def load_results_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            print("Nie udało się wczytać pliku cache. Tworzenie nowego.")
    return {
        'quantum_results': [],
        'quantum_times': {},
        'completed_feature_maps': [],
        'hybrid_scores': {},
        'hybrid_eval_times': {}
    }

# Funkcja do inicjalizacji lokalnego symulatora
def initialize_ibm_quantum():
    print("\n======= INICJALIZACJA LOKALNEGO SYMULATORA =======")
    print("IBM Quantum Cloud wyłączone - używanie lokalnego symulatora")
    
    try:
        backend = Aer.get_backend('qasm_simulator')
        print("✓ Zainicjalizowano lokalny symulator Qiskit Aer")
        print(f"✓ Backend: {backend.name}")
        
        return None, backend, True
        
    except Exception as e:
        print(f"BŁĄD podczas inicjalizacji lokalnego symulatora: {str(e)}")
        return None, None, False

# Funkcja do przygotowania danych
def prepare_data(data_file):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    
    start_time_data = time.time()

    # Wczytanie danych
    data = pd.read_csv(data_file)

    # Wydzielenie zmiennej docelowej
    y = data['Primary_Diagnosis'] 

    # Usunięcie kolumn identyfikacyjnych
    id_columns = ['Project', 'Case_ID']
    data_processed = data.drop(id_columns + ['Grade'], axis=1)

    # Przekształcenie kolumn kategorycznych na binarne
    categorical_columns = data_processed.select_dtypes(include=['object']).columns.tolist()

    # Przekształcenie każdej kolumny kategorycznej
    for col in categorical_columns:
        unique_values = data_processed[col].unique()
        
        if len(unique_values) == 2:
            if set(unique_values) == {'Yes', 'No'}:
                data_processed[col] = data_processed[col].map({'Yes': 1, 'No': 0})
            elif set(unique_values) == {'Male', 'Female'}:
                data_processed[col] = data_processed[col].map({'Male': 1, 'Female': 0})
            elif set(unique_values) == {'GBM', 'LGG'}:
                data_processed[col] = data_processed[col].map({'GBM': 1, 'LGG': 0})
            else:
                data_processed[col] = pd.factorize(data_processed[col])[0]
        else:
            try:
                data_processed[col] = pd.to_numeric(data_processed[col], errors='raise')
            except:
                dummies = pd.get_dummies(data_processed[col], prefix=col, drop_first=True)
                data_processed = pd.concat([data_processed, dummies], axis=1)
                data_processed.drop(col, axis=1, inplace=True)

    # Sprawdzenie, czy wszystkie dane są numeryczne
    non_numeric = data_processed.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        data_processed = data_processed.drop(non_numeric, axis=1)

    # Wypełnienie brakujących wartości
    if data_processed.isnull().sum().sum() > 0:
        data_processed.fillna(data_processed.mean(), inplace=True)

    # Przygotowanie danych do modelowania
    X = data_processed.values

    # Przetwarzanie cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Redukcja wymiarowości

    # Przygotowanie różnych wersji zredukowanych danych
    X_reduced_versions = {}

    if USE_PCA:
        pca_components = min(PCA_COMPONENTS, X_scaled.shape[1])
        pca = PCA(n_components=pca_components)
        X_reduced_pca = pca.fit_transform(X_scaled)
        X_reduced_versions['pca'] = X_reduced_pca
        
    if USE_TSNE:
        tsne = TSNE(
            n_components=TSNE_COMPONENTS,
            perplexity=TSNE_PERPLEXITY,
            learning_rate=TSNE_LEARNING_RATE,
            n_iter=TSNE_MAX_ITER,
            random_state=RANDOM_STATE
        )
        tsne_start_time = time.time()
        X_reduced_tsne = tsne.fit_transform(X_scaled)
        tsne_end_time = time.time()
        tsne_time = tsne_end_time - tsne_start_time
        
        if EVALUATE_SILHOUETTE:
            try:
                silhouette_avg = silhouette_score(X_reduced_tsne, y)
            except Exception as e:
                pass
        
        X_reduced_versions['tsne'] = X_reduced_tsne

    if USE_UMAP:
        umap_reducer = umap.UMAP(
            n_components=UMAP_COMPONENTS,
            n_neighbors=UMAP_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=RANDOM_STATE
        )
        umap_start_time = time.time()
        X_reduced_umap = umap_reducer.fit_transform(X_scaled)
        umap_end_time = time.time()
        umap_time = umap_end_time - umap_start_time
        
        if EVALUATE_SILHOUETTE:
            try:
                silhouette_avg_umap = silhouette_score(X_reduced_umap, y)
            except Exception as e:
                pass
        
        X_reduced_versions['umap'] = X_reduced_umap

    # Wybór domyślnej wersji zredukowanych danych
    if 'pca' in X_reduced_versions:
        X_reduced = X_reduced_versions['pca']
    elif 'umap' in X_reduced_versions:
        X_reduced = X_reduced_versions['umap']
    elif 'tsne' in X_reduced_versions:
        X_reduced = X_reduced_versions['tsne']
    else:
        X_reduced = X_scaled

    # Podział na zbiory treningowy i testowy
    X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
        X_reduced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Podział dla oryginalnych danych
    X_train, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    end_time_data = time.time()
    data_preparation_time = end_time_data - start_time_data

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_reduced': X_train_reduced,
        'X_test_reduced': X_test_reduced,
        'y_train': y_train,
        'y_test': y_test,
        'data_processed': data_processed,
        'preparation_time': data_preparation_time
    }

def run_experiment_parallel(exp_file):
    """Uruchom pojedynczy eksperyment"""
    if exp_file == 'qsvm1_zz.py':
        import qsvm1_zz
        return qsvm1_zz.run_experiment()
    elif exp_file == 'qsvm2_pauli.py':
        import qsvm2_pauli
        return qsvm2_pauli.run_experiment()
    elif exp_file == 'qsvm3_z.py':
        import qsvm3_z
        return qsvm3_z.run_experiment()
    elif exp_file == 'qsvm4_amplitude.py':
        import qsvm4_amplitude
        return qsvm4_amplitude.run_experiment()
    elif exp_file == 'qsvm5_hybrid.py':
        import qsvm5_hybrid
        return qsvm5_hybrid.run_experiment()
    else:
        print(f"Nieznany eksperyment: {exp_file}")
        return None

def run_all_experiments_parallel():
    """Uruchom wszystkie eksperymenty równolegle"""
    experiment_files = [
        'qsvm1_zz.py',
        'qsvm2_pauli.py', 
        'qsvm3_z.py',
        'qsvm4_amplitude.py',
        'qsvm5_hybrid.py'
    ]
    
    # Użyj 5 procesów (jeden na eksperyment)
    with Pool(5) as p:
        results = p.map(run_experiment_parallel, experiment_files)
    
    return results

if __name__ == "__main__":
    print("======= KONTROLER EKSPERYMENTÓW QSVM =======")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"Uruchamianie 5 eksperymentów równolegle...")
    
    results = run_all_experiments_parallel()
    
    print("======= WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE =======")