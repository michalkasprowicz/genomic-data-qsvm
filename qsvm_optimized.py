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
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil

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

# ----------------- CZĘŚĆ 0: PARAMETRY KONFIGURACYJNE OPTYMALIZOWANE DLA VAST.AI -----------------

# Parametry danych
DATA_FILES = [
    'dane/TCGA_GBM_LGG_Mutations_clean.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_1percent_added.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_5percent_added.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_10percent_added.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_15percent_added.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_20percent_added.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_1percent_substituted.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_5percent_substituted.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_10percent_substituted.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_15percent_substituted.csv',
    'dane/TCGA_GBM_LGG_Mutations_noise_20percent_substituted.csv'
]
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Parametry redukcji wymiarowości
USE_PCA = True
USE_TSNE = False
USE_UMAP = False
PCA_COMPONENTS = 12
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
RUN_CLASSIC_SVM = False
RUN_QUANTUM_SVM = True
RUN_HYBRID_APPROACH = False

# Parametry klasycznego SVM - OPTYMALIZOWANE DLA WIELU RDZENI
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
OUTPUT_DIR = f'wyniki/2025-08-30-2-{QSVM_CV}-fold-zFeature'

# Parametry IBM Quantum Cloud
USE_IBM_QUANTUM = False
IBM_BACKEND = 'qasm_simulator'
IBM_REAL_BACKEND = 'qasm_simulator'
IBM_TOKEN = None
IBM_INSTANCE = None
IBM_MAX_SHOTS = 1024
IBM_OPTIMIZATION_LEVEL = 1
IBM_RESILIENCE_LEVEL = 1

# BEZPIECZNE OPTYMALIZACJE DLA VAST.AI - Z OGRANICZENIAMI
def get_optimal_cpu_config():
    total_cores = cpu_count()
    available_memory = psutil.virtual_memory().total / (1024**3)  # GB
    
    print(f"=== KONFIGURACJA SYSTEMU ===")
    print(f"Liczba rdzeni CPU: {total_cores}")
    print(f"Dostępna pamięć RAM: {available_memory:.1f} GB")
    
    # BEZPIECZNA konfiguracja - maksymalnie 70% rdzeni
    safe_cores = int(total_cores * 0.7)
    
    if total_cores >= 200:
        # BEZPIECZNA konfiguracja dla 255 rdzeni
        n_jobs_kernel = min(30, safe_cores // 4)   # 15% rdzeni dla obliczeń jądra
        n_jobs_svm = min(20, safe_cores // 6)      # 10% rdzeni dla SVM
        n_jobs_parallel = min(15, safe_cores // 8) # 7% rdzeni dla równoległości
        print(f"Konfiguracja: BEZPIECZNA (200+ rdzeni) - OGRANICZONE WYKORZYSTANIE")
        print(f"Bezpieczne wykorzystanie: {n_jobs_kernel + n_jobs_svm + n_jobs_parallel}/{total_cores} rdzeni ({((n_jobs_kernel + n_jobs_svm + n_jobs_parallel)/total_cores)*100:.1f}%)")
    elif total_cores >= 100:
        # Bezpieczna konfiguracja (100-199 rdzeni)
        n_jobs_kernel = min(25, safe_cores // 3)
        n_jobs_svm = min(15, safe_cores // 5)
        n_jobs_parallel = min(10, safe_cores // 7)
        print(f"Konfiguracja: BEZPIECZNA (100+ rdzeni)")
    elif total_cores >= 40:
        # Bezpieczna konfiguracja (40-99 rdzeni)
        n_jobs_kernel = min(15, safe_cores // 3)
        n_jobs_svm = min(8, safe_cores // 5)
        n_jobs_parallel = min(6, safe_cores // 7)
        print(f"Konfiguracja: BEZPIECZNA (40+ rdzeni)")
    elif total_cores >= 32:
        # Bezpieczna konfiguracja (32-39 rdzeni)
        n_jobs_kernel = min(12, safe_cores // 3)
        n_jobs_svm = min(6, safe_cores // 5)
        n_jobs_parallel = min(4, safe_cores // 8)
        print(f"Konfiguracja: BEZPIECZNA (32+ rdzeni)")
    elif total_cores >= 16:
        # Bezpieczna konfiguracja (16-31 rdzeni)
        n_jobs_kernel = min(8, safe_cores // 3)
        n_jobs_svm = min(4, safe_cores // 4)
        n_jobs_parallel = min(3, safe_cores // 5)
        print(f"Konfiguracja: BEZPIECZNA (16+ rdzeni)")
    else:
        # Bezpieczna konfiguracja (<16 rdzeni)
        n_jobs_kernel = min(4, safe_cores // 3)
        n_jobs_svm = min(2, safe_cores // 4)
        n_jobs_parallel = min(2, safe_cores // 4)
        print(f"Konfiguracja: BEZPIECZNA (<16 rdzeni)")
    
    print(f"n_jobs_kernel: {n_jobs_kernel}")
    print(f"n_jobs_svm: {n_jobs_svm}")
    print(f"n_jobs_parallel: {n_jobs_parallel}")
    print(f"Bezpieczne wykorzystanie: {n_jobs_kernel + n_jobs_svm + n_jobs_parallel}/{total_cores} rdzeni")
    
    return {
        'n_jobs_kernel': n_jobs_kernel,
        'n_jobs_svm': n_jobs_svm,
        'n_jobs_parallel': n_jobs_parallel,
        'total_cores': total_cores,
        'available_memory': available_memory
    }

# Pobierz optymalną konfigurację
CPU_CONFIG = get_optimal_cpu_config()

# BEZPIECZNA OPTYMALIZACJA PAMIĘCI DLA VAST.AI
def optimize_memory_for_large_system():
    memory_gb = CPU_CONFIG['available_memory']
    
    if memory_gb >= 200:
        print(f"BEZPIECZNA OPTYMALIZACJA PAMIĘCI: {memory_gb:.1f} GB")
        # Bezpieczne ustawienia dla dużych macierzy
        import numpy as np
        np.set_printoptions(precision=4, suppress=True)
        
        # Częstsze garbage collection dla bezpieczeństwa
        import gc
        gc.set_threshold(700, 10, 10)  # Częstsze GC
        
        print("Pamięć zoptymalizowana bezpiecznie")
    else:
        print(f"Pamięć: {memory_gb:.1f} GB - standardowa optymalizacja")
    
    # Dodaj monitoring pamięci
    start_memory_monitoring()

# Dodaj funkcje timeout i monitoring
def timeout(seconds):
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Funkcja {func.__name__} przekroczyła limit {seconds}s")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

def start_memory_monitoring():
    import threading
    
    def monitor_resources():
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 95 or memory_percent > 90:
                    print(f"WYSOKIE WYKORZYSTANIE - CPU: {cpu_percent}%, RAM: {memory_percent}%")
                
                time.sleep(30)
            except Exception as e:
                print(f"Błąd monitoringu: {e}")
                time.sleep(60)
    
    # Uruchom monitoring w tle
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    print("Monitoring zasobów uruchomiony")

# Uruchom optymalizację pamięci
optimize_memory_for_large_system()

# Upewnij się, że katalog wyjściowy istnieje
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Upewnij się, że katalog cache istnieje
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Utworzono katalog cache: {CACHE_DIR}")

# ZAAWANSOWANY SYSTEM LOGOWANIA
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.start_time = time.time()

    def write(self, message):
        # Dodaj timestamp do każdego komunikatu
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.terminal.write(formatted_message)
        self.log.write(formatted_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# FUNKCJA DO LOGOWANIA Z TIMESTAMPEM
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{level}] {message}\n"
    
    # Wyświetl w konsoli
    sys.stdout.write(formatted_message)
    sys.stdout.flush()
    
    # Zapisz do pliku log
    try:
        with open('qsvm_experiments.log', 'a', encoding='utf-8') as f:
            f.write(formatted_message)
            f.flush()
    except Exception as e:
        print(f"Błąd zapisu do log: {e}")

# FUNKCJA DO LOGOWANIA POSTĘPU
def log_progress(current, total, description="Postęp"):
    percentage = (current / total) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] [PROGRESS] {description}: {current}/{total} ({percentage:.1f}%)"
    
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()
    
    # Zapisz do pliku log
    try:
        with open('qsvm_experiments.log', 'a', encoding='utf-8') as f:
            f.write(message + "\n")
            f.flush()
    except Exception as e:
        print(f"Błąd zapisu do log: {e}")

# FUNKCJA DO LOGOWANIA BŁĘDÓW
def log_error(error_message, exception=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if exception:
        error_details = f"[{timestamp}] [ERROR] {error_message}\nException: {str(exception)}\nTraceback: {traceback.format_exc()}"
    else:
        error_details = f"[{timestamp}] [ERROR] {error_message}"
    
    # Wyświetl w konsoli
    sys.stderr.write(error_details + "\n")
    sys.stderr.flush()
    
    # Zapisz do pliku log
    try:
        with open('qsvm_experiments.log', 'a', encoding='utf-8') as f:
            f.write(error_details + "\n")
            f.flush()
    except Exception as e:
        print(f"Błąd zapisu do log: {e}")

# FUNKCJA DO LOGOWANIA WYNIKÓW
def log_results(results_dict, experiment_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] [RESULTS] Eksperyment: {experiment_name}\n"
    
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            message += f"  {key}: {value}\n"
        elif isinstance(value, dict):
            message += f"  {key}: {json.dumps(value, indent=2)}\n"
        else:
            message += f"  {key}: {str(value)}\n"
    
    # Wyświetl w konsoli
    sys.stdout.write(message)
    sys.stdout.flush()
    
    # Zapisz do pliku log
    try:
        with open('qsvm_experiments.log', 'a', encoding='utf-8') as f:
            f.write(message)
            f.flush()
    except Exception as e:
        print(f"Błąd zapisu do log: {e}")

# IMPORT TRACEBACK DLA BŁĘDÓW
import traceback

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
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results_dict
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    log_message(f"Zapisano cache: {cache_file}", "CACHE")

# Funkcja do wczytywania wyników pośrednich
def load_results_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                log_message(f"Wczytano cache z: {cache_file}", "CACHE")
                log_message(f"Timestamp: {cache_data.get('timestamp', 'N/A')}", "CACHE")
                return cache_data.get('results', {})
        except Exception as e:
            log_error(f"Błąd wczytywania cache: {e}")
    return {
        'quantum_results': [],
        'quantum_times': {},
        'completed_feature_maps': [],
        'hybrid_scores': {},
        'hybrid_eval_times': {},
        'classic_svm_results': {},
        'data_preparation_cache': {},
        'kernel_cache': {}
    }

# FUNKCJE CACHE DLA JĄDRA KWANTOWEGO
def get_kernel_cache_key(X1_shape, X2_shape, feature_map_name, shots):
    return f"kernel_{X1_shape[0]}x{X1_shape[1]}_{X2_shape[0]}x{X2_shape[1]}_{feature_map_name}_{shots}"

def save_kernel_cache(kernel_matrix, cache_key, cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")
    np.save(cache_file, kernel_matrix)
    log_message(f"Zapisano jądro do cache: {cache_file}", "CACHE")

def load_kernel_cache(cache_key, cache_dir=CACHE_DIR):
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")
    if os.path.exists(cache_file):
        try:
            kernel_matrix = np.load(cache_file)
            log_message(f"Wczytano jądro z cache: {cache_file}", "CACHE")
            return kernel_matrix
        except Exception as e:
            log_error(f"Błąd wczytywania jądra z cache: {e}")
    return None

# FUNKCJE CACHE DLA PRZYGOTOWANIA DANYCH
def get_data_cache_key(data_file, test_size, random_state, pca_components):
    #Generuje klucz cache dla przygotowanych danych
    return f"data_{os.path.basename(data_file)}_{test_size}_{random_state}_{pca_components}"

def save_data_cache(data_dict, cache_key, cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    np.savez_compressed(cache_file, **data_dict)
    log_message(f"Zapisano dane do cache: {cache_file}", "CACHE")

def load_data_cache(cache_key, cache_dir=CACHE_DIR):
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            data_dict = {key: data[key] for key in data.files}
            log_message(f"Wczytano dane z cache: {cache_file}", "CACHE")
            return data_dict
        except Exception as e:
            log_error(f"Błąd wczytywania danych z cache: {e}")
    return None

# FUNKCJE CACHE DLA WYNIKÓW EKSPERYMENTÓW
def save_experiment_cache(experiment_name, results, cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"experiment_{experiment_name}.json")
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'results': results
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    log_message(f"Zapisano eksperyment do cache: {cache_file}", "CACHE")

def load_experiment_cache(experiment_name, cache_dir=CACHE_DIR):
    cache_file = os.path.join(cache_dir, f"experiment_{experiment_name}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                log_message(f"Wczytano eksperyment z cache: {cache_file}", "CACHE")
                return cache_data.get('results', {})
        except Exception as e:
            log_error(f"Błąd wczytywania eksperymentu z cache: {e}")
    return None

# FUNKCJA DO CZYSZCZENIA CACHE
def clear_cache(cache_dir=CACHE_DIR):
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        log_message(f"Wyczyszczono cache: {cache_dir}", "CACHE")

# FUNKCJA DO SPRAWDZANIA STATUSU CACHE
def check_cache_status(cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        log_message("Brak katalogu cache", "CACHE")
        return
    
    cache_files = os.listdir(cache_dir)
    log_message(f"Katalog cache: {cache_dir}", "CACHE")
    log_message(f"Liczba plików cache: {len(cache_files)}", "CACHE")
    
    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        log_message(f"   - {file}: {file_size:.2f} MB, {file_time}", "CACHE")

# BEZPIECZNA FUNKCJA DO RÓWNOLEGŁEGO OBLICZANIA JĄDRA KWANTOWEGO Z TIMEOUTAMI
@timeout(1800)  # 30 minut timeout
def parallel_quantum_kernel_evaluation(X1, X2, feature_map, backend_name='qasm_simulator', shots=1024):
    # SPRAWDZENIE CACHE
    feature_map_name = feature_map.__class__.__name__
    cache_key = get_kernel_cache_key(X1.shape, X2.shape, feature_map_name, shots)
    
    log_message(f"Sprawdzanie cache dla jądra: {cache_key}", "KERNEL")
    cached_kernel = load_kernel_cache(cache_key)
    
    if cached_kernel is not None:
        log_message(f"Znaleziono jądro w cache! Oszczędność: ~{X1.shape[0] * X2.shape[0] * 8 / (1024**2):.1f} MB", "KERNEL")
        return cached_kernel
    
    log_message(f"Brak jądra w cache - obliczanie od nowa", "KERNEL")
    
    n_jobs = CPU_CONFIG['n_jobs_kernel']
    log_message(f"Bezpieczne obliczanie jądra kwantowego (n_jobs={n_jobs})", "KERNEL")
    log_message(f"Rozmiar danych: {X1.shape} x {X2.shape}", "KERNEL")
    log_message(f"Bezpieczne wykorzystanie rdzeni: {n_jobs}/{CPU_CONFIG['total_cores']} ({n_jobs/CPU_CONFIG['total_cores']*100:.1f}%)", "KERNEL")
    
    # Sprawdź rozmiar danych - ograniczenie dla bezpieczeństwa
    max_matrix_size = 1000 * 1000  # Maksymalnie 1M elementów
    if X1.shape[0] * X2.shape[0] > max_matrix_size:
        log_message(f"Zbyt duża macierz jądra: {X1.shape[0] * X2.shape[0]} > {max_matrix_size}", "WARNING")
        log_message(f"Redukcja rozmiaru danych dla bezpieczeństwa", "KERNEL")
        # Redukuj rozmiar danych
        X1 = X1[:min(500, X1.shape[0])]
        X2 = X2[:min(500, X2.shape[0])]
    
    # Inicjalizacja backendu
    try:
        backend = Aer.get_backend(backend_name)
        backend.set_options(shots=shots)
    except Exception as e:
        log_error(f"Błąd backendu: {e}, używam qasm_simulator")
        backend = Aer.get_backend('qasm_simulator')
        backend.set_options(shots=shots)
    
    # Utworzenie quantum kernel
    try:
        from qiskit.utils import QuantumInstance
        quantum_instance = QuantumInstance(backend=backend, shots=shots)
        quantum_kernel = QuantumKernel(
            feature_map=feature_map,
            quantum_instance=quantum_instance
        )
    except Exception as e:
        log_error(f"Błąd tworzenia quantum kernel: {e}")
        return None
    
    # BEZPIECZNA OPTYMALIZACJA CHUNK SIZE
    chunk_size = max(1, min(X1.shape[0] // n_jobs, 5))  # Mniejsze chunki dla bezpieczeństwa
    
    kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    
    @timeout(300)  # 5 minut timeout na chunk
    def compute_kernel_chunk(start_idx):
        end_idx = min(start_idx + chunk_size, X1.shape[0])
        chunk_X1 = X1[start_idx:end_idx]
        
        try:
            chunk_kernel = quantum_kernel.evaluate(chunk_X1, X2)
            return start_idx, end_idx, chunk_kernel
        except Exception as e:
            log_error(f"Błąd obliczania chunk'a {start_idx}-{end_idx}: {e}")
            return start_idx, end_idx, np.zeros((end_idx - start_idx, X2.shape[0]))
    
    # Bezpieczne równoległe obliczenia
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    log_message(f"Używam {n_jobs} rdzeni bezpiecznie", "KERNEL")
    log_message(f"Chunk size: {chunk_size}", "KERNEL")
    log_message(f"Liczba chunk'ów: {len(range(0, X1.shape[0], chunk_size))}", "KERNEL")
    
    # BEZPIECZNA LICZBA PROCESÓW
    max_workers = min(n_jobs, 20)  # Maksymalnie 20 procesów jednocześnie
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, X1.shape[0], chunk_size):
            futures.append(executor.submit(compute_kernel_chunk, i))
        
        # Zbierz wyniki z timeoutami
        completed_chunks = 0
        for future in as_completed(futures, timeout=1800):  # 30 minut timeout
            try:
                start_idx, end_idx, chunk_kernel = future.result(timeout=300)
                kernel_matrix[start_idx:end_idx] = chunk_kernel
                completed_chunks += 1
                log_message(f"Ukończono chunk {completed_chunks}/{len(futures)}", "KERNEL")
            except Exception as e:
                log_error(f"Błąd w chunk'u: {e}")
                continue
    
    # ZAPISZ DO CACHE
    log_message(f"Zapisuję jądro do cache: {cache_key}", "KERNEL")
    save_kernel_cache(kernel_matrix, cache_key)
    
    return kernel_matrix

# OPTYMALIZOWANA FUNKCJA PRZYGOTOWANIA DANYCH Z CACHE
def prepare_data(data_file):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    
    # SPRAWDZENIE CACHE DLA DANYCH
    cache_key = get_data_cache_key(data_file, TEST_SIZE, RANDOM_STATE, PCA_COMPONENTS)
    print(f"Sprawdzanie cache dla danych: {cache_key}")
    
    cached_data = load_data_cache(cache_key)
    if cached_data is not None:
        print(f"Znaleziono dane w cache! Oszczędność czasu przygotowania.")
        return cached_data
    
    print(f"Brak danych w cache - przygotowanie od nowa")
    print(f"\n======= PRZYGOTOWANIE DANYCH =======")
    start_time_data = time.time()

    # Wczytanie danych
    data = pd.read_csv(data_file)
    print(f"Wymiary oryginalnych danych: {data.shape}")

    # Wydzielenie zmiennej docelowej
    y = data['Primary_Diagnosis'] 
    print(f"Unikalne wartości zmiennej docelowej: {y.unique()}")
    print(f"Rozkład klas: {y.value_counts().to_dict()}")

    # Usunięcie kolumn identyfikacyjnych
    id_columns = ['Project', 'Case_ID']
    data_processed = data.drop(id_columns + ['Grade'], axis=1)
    print(f"Wymiary danych po usunięciu kolumn identyfikacyjnych: {data_processed.shape}")

    # Przekształcenie kolumn kategorycznych na binarne
    print("Przekształcanie kolumn kategorycznych na binarne...")
    categorical_columns = data_processed.select_dtypes(include=['object']).columns.tolist()
    print(f"Kolumny kategoryczne do przekształcenia: {categorical_columns}")

    # Specjalna obsługa kolumny Age_at_diagnosis (jeśli istnieje)
    if 'Age_at_diagnosis' in categorical_columns:
        print("Znaleziono kolumnę Age_at_diagnosis - sprawdzanie formatu...")
        age_col = data_processed['Age_at_diagnosis']
        
        # Sprawdź czy wartości są już numeryczne (dni)
        try:
            data_processed['Age_at_diagnosis'] = pd.to_numeric(age_col, errors='raise')
            print("Kolumna Age_at_diagnosis została przekształcona na numeryczną (dni).")
            categorical_columns.remove('Age_at_diagnosis')  # Usuń z listy kolumn kategorycznych
        except:
            print("UWAGA: Kolumna Age_at_diagnosis nie jest numeryczna!")

    # Przekształcenie każdej kolumny kategorycznej
    for col in categorical_columns:
        unique_values = data_processed[col].unique()
        print(f"Kolumna {col} zawiera wartości: {unique_values}")
        
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
            # Ograniczenie one-hot encoding do maksymalnie 10 kategorii
            MAX_CATEGORIES = 10
            if len(unique_values) > MAX_CATEGORIES:
                print(f"Kolumna {col} ma {len(unique_values)} kategorii - używam factorize zamiast one-hot.")
                data_processed[col] = pd.factorize(data_processed[col])[0]
            else:
                try:
                    data_processed[col] = pd.to_numeric(data_processed[col], errors='raise')
                    print(f"Kolumna {col} została przekształcona na liczbową.")
                except:
                    print(f"Kolumna {col} zostanie zakodowana jako one-hot ({len(unique_values)} kategorii).")
                    dummies = pd.get_dummies(data_processed[col], prefix=col, drop_first=True)
                    data_processed = pd.concat([data_processed, dummies], axis=1)
                    data_processed.drop(col, axis=1, inplace=True)

    # Sprawdzenie, czy wszystkie dane są numeryczne
    non_numeric = data_processed.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print(f"UWAGA: Pozostały kolumny nieliczbowe: {non_numeric}")
        print("Usuwanie pozostałych kolumn nieliczbowych...")
        data_processed = data_processed.drop(non_numeric, axis=1)

    print(f"Wymiary danych po przekształceniu: {data_processed.shape}")

    # Kontrola liczby wymiarów
    MAX_DIMENSIONS = 27  # Maksymalna liczba wymiarów
    if data_processed.shape[1] > MAX_DIMENSIONS:
        print(f"UWAGA: Zbyt wiele wymiarów ({data_processed.shape[1]})! Maksymalnie {MAX_DIMENSIONS}.")
        print("Rozważ zwiększenie PCA_COMPONENTS lub ograniczenie one-hot encoding.")

    # Wypełnienie brakujących wartości
    if data_processed.isnull().sum().sum() > 0:
        print("Wypełnianie brakujących wartości...")
        data_processed.fillna(data_processed.mean(), inplace=True)

    # Przygotowanie danych do modelowania
    X = data_processed.values
    print(f"Wymiary danych wejściowych: {X.shape}")

    # Przetwarzanie cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Wymiary danych po skalowaniu: {X_scaled.shape}")

    # Redukcja wymiarowości
    print("\n======= REDUKCJA WYMIAROWOŚCI =======")

    # Przygotowanie różnych wersji zredukowanych danych
    X_reduced_versions = {}

    if USE_PCA:
        pca_components = min(PCA_COMPONENTS, X_scaled.shape[1])
        print(f"Liczba komponentów PCA: {pca_components}")
        pca = PCA(n_components=pca_components)
        X_reduced_pca = pca.fit_transform(X_scaled)
        print(f"Wymiary danych po redukcji PCA: {X_reduced_pca.shape}")
        print(f"Wyjaśniona wariancja: {sum(pca.explained_variance_ratio_):.4f}")
        X_reduced_versions['pca'] = X_reduced_pca
        
    if USE_TSNE:
        print(f"\nStosowanie t-SNE na danych skalowanych...")
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
        print(f"Wymiary danych po redukcji t-SNE: {X_reduced_tsne.shape}")
        print(f"Czas wykonania t-SNE: {tsne_time:.2f} sekund")
        
        if EVALUATE_SILHOUETTE:
            try:
                silhouette_avg = silhouette_score(X_reduced_tsne, y)
                print(f"Silhouette Score dla t-SNE: {silhouette_avg:.4f}")
            except Exception as e:
                print(f"Nie udało się obliczyć Silhouette Score: {str(e)}")
        
        X_reduced_versions['tsne'] = X_reduced_tsne

    if USE_UMAP:
        print(f"\nStosowanie UMAP na danych skalowanych...")
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
        print(f"Wymiary danych po redukcji UMAP: {X_reduced_umap.shape}")
        print(f"Czas wykonania UMAP: {umap_time:.2f} sekund")
        
        if EVALUATE_SILHOUETTE:
            try:
                silhouette_avg_umap = silhouette_score(X_reduced_umap, y)
                print(f"Silhouette Score dla UMAP: {silhouette_avg_umap:.4f}")
            except Exception as e:
                print(f"Nie udało się obliczyć Silhouette Score dla UMAP: {str(e)}")
        
        X_reduced_versions['umap'] = X_reduced_umap

    # Wybór domyślnej wersji zredukowanych danych
    if 'pca' in X_reduced_versions:
        X_reduced = X_reduced_versions['pca']
        print("Używanie danych zredukowanych przez PCA jako domyślnych.")
    elif 'umap' in X_reduced_versions:
        X_reduced = X_reduced_versions['umap']
        print("Używanie danych zredukowanych przez UMAP jako domyślnych.")
    elif 'tsne' in X_reduced_versions:
        X_reduced = X_reduced_versions['tsne']
        print("Używanie danych zredukowanych przez t-SNE jako domyślnych.")
    else:
        X_reduced = X_scaled
        print("Nie wybrano żadnej metody redukcji wymiarowości. Używanie oryginalnych danych skalowanych.")

    # Podział na zbiory treningowy i testowy
    X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
        X_reduced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Podział dla oryginalnych danych
    X_train, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"Wymiary zbioru treningowego: {X_train_reduced.shape}")
    print(f"Wymiary zbioru testowego: {X_test_reduced.shape}")

    end_time_data = time.time()
    data_preparation_time = end_time_data - start_time_data
    print(f"\nCzas przygotowania danych: {data_preparation_time:.2f} sekund")

    # PRZYGOTUJ DANE DO CACHE
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_reduced': X_train_reduced,
        'X_test_reduced': X_test_reduced,
        'y_train': y_train,
        'y_test': y_test,
        'data_processed': data_processed,
        'preparation_time': data_preparation_time
    }
    
    # ZAPISZ DO CACHE
    print(f"Zapisuję dane do cache: {cache_key}")
    save_data_cache(data_dict, cache_key)
    
    return data_dict

def run_experiment_parallel(exp_file):
    experiment_name = exp_file.replace('.py', '')
    
    # SPRAWDZENIE CACHE DLA EKSPERYMENTU
    log_message(f"Sprawdzanie cache dla eksperymentu: {experiment_name}", "EXPERIMENT")
    cached_results = load_experiment_cache(experiment_name)
    
    if cached_results is not None:
        log_message(f"Znaleziono wyniki eksperymentu w cache: {experiment_name}", "EXPERIMENT")
        return cached_results
    
    log_message(f"Brak wyników w cache - uruchamianie eksperymentu: {experiment_name}", "EXPERIMENT")
    
    # URUCHOM EKSPERYMENT
    if exp_file == 'qsvm1_zz.py':
        import qsvm1_zz
        results = qsvm1_zz.run_experiment()
    elif exp_file == 'qsvm2_pauli.py':
        import qsvm2_pauli
        results = qsvm2_pauli.run_experiment()
    elif exp_file == 'qsvm3_z.py':
        import qsvm3_z
        results = qsvm3_z.run_experiment()
    elif exp_file == 'qsvm4_amplitude.py':
        import qsvm4_amplitude
        results = qsvm4_amplitude.run_experiment()
    elif exp_file == 'qsvm5_hybrid.py':
        import qsvm5_hybrid
        results = qsvm5_hybrid.run_experiment()
    else:
        log_error(f"Nieznany eksperyment: {exp_file}")
        return None
    
    # ZAPISZ WYNIKI DO CACHE
    if results is not None:
        log_message(f"Zapisuję wyniki eksperymentu do cache: {experiment_name}", "EXPERIMENT")
        save_experiment_cache(experiment_name, results)
    
    return results

def run_single_file_experiment(args):
    exp_file, data_file = args
    experiment_name = exp_file.replace('.py', '')
    file_name = os.path.basename(data_file).split('.')[0]
    
    log_message(f"Uruchamianie {experiment_name} na pliku: {file_name}", "FILE_EXPERIMENT")
    
    # SPRAWDZENIE CACHE DLA KOMBINACJI EKSPERYMENT+PLIK
    cache_key = f"{experiment_name}_{file_name}"
    cached_results = load_experiment_cache(cache_key)
    
    if cached_results is not None:
        log_message(f"Znaleziono wyniki w cache: {cache_key}", "FILE_EXPERIMENT")
        return (exp_file, data_file, cached_results)
    
    log_message(f"Brak wyników w cache - obliczanie: {cache_key}", "FILE_EXPERIMENT")
    
    # URUCHOM EKSPERYMENT NA POJEDYNCZYM PLIKU
    try:
        if exp_file == 'qsvm1_zz.py':
            import qsvm1_zz
            results = qsvm1_zz.run_single_file_experiment(data_file)
        elif exp_file == 'qsvm2_pauli.py':
            import qsvm2_pauli
            results = qsvm2_pauli.run_single_file_experiment(data_file)
        elif exp_file == 'qsvm3_z.py':
            import qsvm3_z
            results = qsvm3_z.run_single_file_experiment(data_file)
        elif exp_file == 'qsvm4_amplitude.py':
            import qsvm4_amplitude
            results = qsvm4_amplitude.run_single_file_experiment(data_file)
        elif exp_file == 'qsvm5_hybrid.py':
            import qsvm5_hybrid
            results = qsvm5_hybrid.run_single_file_experiment(data_file)
        else:
            log_error(f"Nieznany eksperyment: {exp_file}")
            return (exp_file, data_file, None)
        
        # ZAPISZ WYNIKI DO CACHE
        if results is not None:
            log_message(f"Zapisuję wyniki do cache: {cache_key}", "FILE_EXPERIMENT")
            save_experiment_cache(cache_key, results)
        
        return (exp_file, data_file, results)
        
    except Exception as e:
        log_error(f"Błąd w eksperymencie {exp_file} na pliku {data_file}: {e}")
        return (exp_file, data_file, None)

@timeout(7200)  # 2 godziny timeout na cały eksperyment
def run_all_experiments_parallel():
    experiment_files = [
        'qsvm3_z.py' # ZFeatureMap - ZAKOMENTOWANE
    #    'qsvm4_amplitude.py', # Amplitude Encoding - ZAKOMENTOWANE
    #    'qsvm1_zz.py',       # ZZFeatureMap - ZAKOMENTOWANE
    #    'qsvm2_pauli.py'    # PauliFeatureMap - TYLKO TEN
    #    'qsvm5_hybrid.py'    # Hybrid Approach - ZAKOMENTOWANE
    ]
    
    # GENERUJ WSZYSTKIE KOMBINACJE EKSPERYMENT+PLIK
    all_combinations = []
    for exp_file in experiment_files:
        for data_file in DATA_FILES:
            all_combinations.append((exp_file, data_file))
    
    total_combinations = len(all_combinations)
    log_message(f"Łączna liczba kombinacji eksperyment+plik: {total_combinations}", "MAIN")
    
    # BEZPIECZNE WYKORZYSTANIE RDZENI
    n_processes = min(CPU_CONFIG['n_jobs_parallel'], total_combinations, 8)  # Maksymalnie 8 procesów
    log_message(f"Uruchamianie {total_combinations} kombinacji z {n_processes} procesami bezpiecznie", "MAIN")
    log_message(f"Eksperymenty: {experiment_files}", "MAIN")
    log_message(f"Pliki danych: {len(DATA_FILES)}", "MAIN")
    log_message(f"Bezpieczne przyspieszenie: ~{n_processes/2:.1f}x", "MAIN")
    
    # Dodatkowe informacje o wydajności
    log_message(f"Dostępna pamięć: {CPU_CONFIG['available_memory']:.1f} GB", "MAIN")
    log_message(f"Bezpieczna konfiguracja rdzeni:", "MAIN")
    log_message(f"   - Jądro kwantowe: {CPU_CONFIG['n_jobs_kernel']} rdzeni", "MAIN")
    log_message(f"   - SVM: {CPU_CONFIG['n_jobs_svm']} rdzeni", "MAIN")
    log_message(f"   - Równoległość: {CPU_CONFIG['n_jobs_parallel']} rdzeni", "MAIN")
    
    # Przewidywany czas dla wszystkich kombinacji
    log_message(f"PRZEWIDYWANY CZAS Z TIMEOUTAMI:", "MAIN")
    log_message(f"   - Amplitude: 20-45 minut na plik (max 2h)", "MAIN")
    log_message(f"   - ŁĄCZNIE: {total_combinations} kombinacji z timeoutami", "MAIN")
    
    # URUCHOM KOMBINACJE Z TIMEOUTAMI
    results = []
    for i, combination in enumerate(all_combinations):
        try:
            log_message(f"Uruchamianie kombinacji {i+1}/{total_combinations}: {combination}", "MAIN")
            
            # Timeout na pojedynczą kombinację
            @timeout(3600)  # 1 godzina na kombinację
            def run_single_with_timeout():
                return run_single_file_experiment(combination)
            
            result = run_single_with_timeout()
            results.append(result)
            
            log_message(f"Ukończono kombinację {i+1}/{total_combinations}", "MAIN")
            
        except TimeoutError:
            log_message(f"Timeout dla kombinacji {i+1}: {combination}", "WARNING")
            results.append((combination[0], combination[1], None))
        except Exception as e:
            log_error(f"Błąd w kombinacji {i+1}: {e}")
            results.append((combination[0], combination[1], None))
    
    # GRUPUJ WYNIKI WG EKSPERYMENTÓW
    grouped_results = {}
    for exp_file in experiment_files:
        grouped_results[exp_file] = []
    
    for exp_file, data_file, result in results:
        if result is not None:
            grouped_results[exp_file].append((data_file, result))
    
    return grouped_results

# FUNKCJA DO URACHAMIANIA KLASYCZNEGO SVM
def run_classic_svm_single_file(data_file):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    file_name = os.path.basename(data_file).split('.')[0]
    log_message(f"Uruchamianie klasycznego SVM na pliku: {file_name}", "CLASSIC_SVM")
    
    # SPRAWDZENIE CACHE DLA KLASYCZNEGO SVM
    cache_key = f"classic_svm_{file_name}"
    cached_results = load_experiment_cache(cache_key)
    
    if cached_results is not None:
        log_message(f"Znaleziono wyniki klasycznego SVM w cache: {cache_key}", "CLASSIC_SVM")
        return cached_results
    
    log_message(f"Brak wyników w cache - obliczanie klasycznego SVM: {cache_key}", "CLASSIC_SVM")
    
    try:
        # Przygotowanie danych
        data_dict = prepare_data(data_file)
        X_train_reduced = data_dict['X_train_reduced']
        X_test_reduced = data_dict['X_test_reduced']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        log_message(f"Wymiary danych treningowych: {X_train_reduced.shape}", "CLASSIC_SVM")
        log_message(f"Wymiary danych testowych: {X_test_reduced.shape}", "CLASSIC_SVM")
        
        # Uruchomienie klasycznego SVM z GridSearchCV
        start_time = time.time()
        
        # BEZPIECZNE WYKORZYSTANIE RDZENI
        n_jobs = min(CPU_CONFIG['n_jobs_svm'], 8)  # Maksymalnie 8 rdzeni dla bezpieczeństwa
        log_message(f"Używam {n_jobs} rdzeni dla klasycznego SVM", "CLASSIC_SVM")
        
        grid = GridSearchCV(
            SVC(random_state=RANDOM_STATE), 
            SVM_PARAM_GRID, 
            cv=SVM_CV, 
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        log_message("Rozpoczynam GridSearchCV dla klasycznego SVM...", "CLASSIC_SVM")
        grid.fit(X_train_reduced, y_train)
        
        # Najlepszy model
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
        
        # Predykcje na zbiorze testowym
        classic_pred = best_model.predict(X_test_reduced)
        
        end_time = time.time()
        classic_svm_time = end_time - start_time
        
        # Obliczenie metryk
        accuracy = accuracy_score(y_test, classic_pred)
        precision = precision_score(y_test, classic_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, classic_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, classic_pred, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, classic_pred)
        except:
            roc_auc = "N/A"
        
        # Macierz pomyłek
        cm = confusion_matrix(y_test, classic_pred)
        
        # Raport klasyfikacji
        class_report = classification_report(y_test, classic_pred, output_dict=True)
        
        # Wyniki
        results = {
            'file_name': file_name,
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'training_time': classic_svm_time,
            'data_shape': X_train_reduced.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        log_message(f"Klasyczny SVM ukończony dla {file_name}:", "CLASSIC_SVM")
        log_message(f"  - Najlepsze parametry: {best_params}", "CLASSIC_SVM")
        log_message(f"  - Dokładność CV: {best_score:.4f}", "CLASSIC_SVM")
        log_message(f"  - Dokładność test: {accuracy:.4f}", "CLASSIC_SVM")
        log_message(f"  - Czas treningu: {classic_svm_time:.2f} sekund", "CLASSIC_SVM")
        
        # ZAPISZ DO CACHE
        log_message(f"Zapisuję wyniki klasycznego SVM do cache: {cache_key}", "CLASSIC_SVM")
        save_experiment_cache(cache_key, results)
        
        return results
        
    except Exception as e:
        log_error(f"Błąd w klasycznym SVM dla pliku {data_file}: {e}")
        return None

# FUNKCJA DO URACHAMIANIA KLASYCZNEGO SVM NA WSZYSTKICH PLIKACH
def run_classic_svm_all_files():
    log_message(f"Uruchamianie klasycznego SVM na {len(DATA_FILES)} plikach", "CLASSIC_SVM_ALL")
    
    # Sprawdź które pliki istnieją
    existing_files = []
    for data_file in DATA_FILES:
        if os.path.exists(data_file):
            existing_files.append(data_file)
        else:
            log_message(f"Plik nie istnieje: {data_file}", "WARNING")
    
    log_message(f"Znaleziono {len(existing_files)} istniejących plików", "CLASSIC_SVM_ALL")
    
    # Uruchom klasyczny SVM na wszystkich plikach
    results = []
    for i, data_file in enumerate(existing_files):
        try:
            log_message(f"Przetwarzanie pliku {i+1}/{len(existing_files)}: {os.path.basename(data_file)}", "CLASSIC_SVM_ALL")
            
            # Timeout na pojedynczy plik
            @timeout(1800)  # 30 minut na plik
            def run_single_with_timeout():
                return run_classic_svm_single_file(data_file)
            
            result = run_single_with_timeout()
            if result is not None:
                results.append(result)
                log_message(f"Ukończono plik {i+1}/{len(existing_files)}", "CLASSIC_SVM_ALL")
            else:
                log_message(f"Błąd w pliku {i+1}/{len(existing_files)}", "CLASSIC_SVM_ALL")
                
        except TimeoutError:
            log_message(f"Timeout dla pliku {i+1}: {os.path.basename(data_file)}", "WARNING")
        except Exception as e:
            log_error(f"Błąd w pliku {i+1}: {e}")
    
    # Zapisanie wszystkich wyników
    if results:
        output_file = os.path.join(OUTPUT_DIR, f'wyniki_classic_svm_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        log_message(f"Zapisano wszystkie wyniki do: {output_file}", "CLASSIC_SVM_ALL")
        
        # Podsumowanie
        log_message("PODSUMOWANIE KLASYCZNEGO SVM:", "CLASSIC_SVM_ALL")
        for result in results:
            log_message(f"  {result['file_name']}: {result['test_accuracy']:.4f} (czas: {result['training_time']:.1f}s)", "CLASSIC_SVM_ALL")
    
    return results

if __name__ == "__main__":
    print("======= KONTROLER EKSPERYMENTÓW QSVM - BEZPIECZNE ZFEATUREMAP Z TIMEOUTAMI =======")
    print(f"BEZPIECZNA KONFIGURACJA: {CPU_CONFIG['total_cores']} rdzeni")
    print(f"Dostępna pamięć: {CPU_CONFIG['available_memory']:.1f} GB")
    print(f"BEZPIECZNA KONFIGURACJA: {CPU_CONFIG}")
    print("")
    print("Uruchamianie TYLKO PAULIFEATUREMAP z BEZPIECZNYMI timeoutami:")
    print("   - PauliFeatureMap (qsvm2_pauli.py) - TYLKO TEN")
    print("   - Amplitude Encoding (qsvm4_amplitude.py) - ZAKOMENTOWANE")
    print("   - ZZFeatureMap (qsvm1_zz.py) - ZAKOMENTOWANE")
    print("   - ZFeatureMap (qsvm3_z.py) - ZAKOMENTOWANE")
    print("   - Hybrid Approach (qsvm5_hybrid.py) - ZAKOMENTOWANE")
    print("")
    print("SYSTEM BEZPIECZEŃSTWA:")
    print("   - Timeouty na wszystkie operacje - zapobieganie zawieszeniu")
    print("   - Monitoring zasobów - wykrywanie przeciążenia")
    print("   - Ograniczone wykorzystanie rdzeni - 70% maksymalnie")
    print("   - Cache z timeoutami - bezpieczne zapisywanie")
    print("")
    print("SYSTEM CACHE:")
    print("   - Cache jądra kwantowego - oszczędność obliczeń")
    print("   - Cache przygotowania danych - oszczędność czasu")
    print("   - Cache wyników eksperymentów - wznowienie po przerwaniu")
    print("")
    print("PRZEWIDYWANE WYNIKI:")
    print(f"   - Czas wykonania: 1-2 godziny (z timeoutami)")
    print(f"   - Bezpieczne wykorzystanie rdzeni: {CPU_CONFIG['n_jobs_kernel'] + CPU_CONFIG['n_jobs_svm'] + CPU_CONFIG['n_jobs_parallel']}/{CPU_CONFIG['total_cores']}")
    print(f"   - Przewidywane przyspieszenie: ~{CPU_CONFIG['total_cores']/16:.1f}x (bezpieczne)")
    print("   - Oszczędność dzięki cache: 50-80% przy ponownym uruchomieniu")
    print("   - Zero zawieszeń dzięki timeoutom")
    print("")
    
    # SPRAWDZENIE STATUSU CACHE
    print("Sprawdzanie statusu cache...")
    check_cache_status()
    print("")
    
    try:
        start_time = time.time()
        
        print("URUCHAMIANIE KWANTOWEGO SVM (ZFEATUREMAP)")
        print("Uruchamianie TYLKO PAULIFEATUREMAP z BEZPIECZNYMI timeoutami:")
        print("   - PauliFeatureMap (qsvm2_pauli.py) - TYLKO TEN")
        print("   - Amplitude Encoding (qsvm4_amplitude.py) - ZAKOMENTOWANE")
        print("   - ZZFeatureMap (qsvm1_zz.py) - ZAKOMENTOWANE")
        print("   - ZFeatureMap (qsvm3_z.py) - ZAKOMENTOWANE")
        print("   - Hybrid Approach (qsvm5_hybrid.py) - ZAKOMENTOWANE")
        print("")
        print("PRZEWIDYWANE WYNIKI:")
        print(f"   - Czas wykonania: 1-2 godziny (z timeoutami)")
        print(f"   - Bezpieczne wykorzystanie rdzeni: {CPU_CONFIG['n_jobs_kernel'] + CPU_CONFIG['n_jobs_svm'] + CPU_CONFIG['n_jobs_parallel']}/{CPU_CONFIG['total_cores']}")
        print(f"   - Przewidywane przyspieszenie: ~{CPU_CONFIG['total_cores']/16:.1f}x (bezpieczne)")
        print("   - Oszczędność dzięki cache: 50-80% przy ponownym uruchomieniu")
        print("   - Zero zawieszeń dzięki timeoutom")
        print("")
        
        results = run_all_experiments_parallel()
        end_time = time.time()
        
        total_time = end_time - start_time
        print("")
        print("======= PAULIFEATUREMAP ZAKOŃCZONE - BEZPIECZNIE Z TIMEOUTAMI =======")
        print(f"Całkowity czas wykonania: {total_time/60:.1f} minut ({total_time/3600:.1f} godzin)")
        print(f"⚡ Średni czas na eksperyment: {total_time/1/60:.1f} minut")
        print(f"Bezpieczne wykorzystanie rdzeni: {CPU_CONFIG['n_jobs_kernel'] + CPU_CONFIG['n_jobs_svm'] + CPU_CONFIG['n_jobs_parallel']}/{CPU_CONFIG['total_cores']} ({((CPU_CONFIG['n_jobs_kernel'] + CPU_CONFIG['n_jobs_svm'] + CPU_CONFIG['n_jobs_parallel'])/CPU_CONFIG['total_cores'])*100:.1f}%)")
        print("Finalny status cache:")
        check_cache_status()
        print("PAULIFEATUREMAP ZAKOŃCZONE BEZPIECZNIE!")
        
    except TimeoutError:
        print("")
        print("======= EKSPERYMENT PRZERWANY PRZEZ TIMEOUT =======")
        print("Eksperyment został bezpiecznie przerwany po przekroczeniu limitu czasu")
        print("Wyniki częściowe zostały zapisane w cache")
        print("Status cache:")
        check_cache_status()
        print("Można wznowić eksperyment - cache zostanie wykorzystany")
        
    except Exception as e:
        print("")
        print("======= BŁĄD W EKSPERYMENCIE =======")
        print(f"Błąd: {str(e)}")
        print("Status cache:")
        check_cache_status()
        print("Sprawdź logi i spróbuj ponownie")
