# Kompleksowy eksperyment do pracy magisterskiej - NAPRAWIONA WERSJA
# 
# NAPRAWY:
# 1. Dodano timeout'y dla operacji kwantowych (5 minut)
# 2. Zmniejszono liczbę shots z 1024 na 50
# 3. Ograniczono wymiar cech do maksymalnie 8
# 4. Wyłączono równoległe przetwarzanie (problematyczne)
# 5. Dodano checkpoint'ing co 5 iteracji
# 6. Uproszczono mapy cech - tylko proste warianty
# 7. Dodano lepszą obsługę błędów i logging
# 8. Wyłączono problematyczne eksperymenty
# 
# INSTRUKCJE UŻYCIA:
# - Uruchom na serwerze w chmurze
# - Monitoruj logi w czasie rzeczywistym
# - Eksperyment może trwać 2-4 godziny (zamiast 30+)
# - Wyniki są zapisywane automatycznie
import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  # GridSearchCV, cross_val_score - nie używane
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score  # classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score - nie używane
# from sklearn.inspection import permutation_importance  # Nie używane
from sklearn.decomposition import PCA
# import json  # Nie używane
# import os.path  # Nie używane
import signal
# import multiprocessing as mp  # Nie używane
# from functools import partial  # Nie używane
import warnings
warnings.filterwarnings('ignore')

# Import bibliotek kwantowych - NAPRAWIONE IMPORTY
import qiskit
# Naprawiony import Aer dla nowszych wersji Qiskit
try:
    from qiskit import Aer
    print("✓ Import Aer z qiskit - sukces")
except ImportError:
    try:
        from qiskit_aer import Aer
        print("✓ Import Aer z qiskit_aer - sukces")
    except ImportError:
        try:
            from qiskit.providers.aer import Aer
            print("✓ Import Aer z qiskit.providers.aer - sukces")
        except ImportError:
            print("❌ Błąd: Nie można zaimportować Aer z żadnego źródła")
            print("Sprawdź czy qiskit-aer jest zainstalowany: pip install qiskit-aer")
            sys.exit(1)

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2
from qiskit.circuit.library.data_preparation import ZFeatureMap

# Naprawiony import QuantumKernel dla nowszych wersji
try:
    from qiskit_machine_learning.kernels import QuantumKernel
    print("✓ Import QuantumKernel z qiskit_machine_learning.kernels - sukces")
except ImportError:
    try:
        from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
        print("✓ Import QuantumKernel z qiskit_machine_learning.kernels.quantum_kernel - sukces")
    except ImportError:
        try:
            from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
            print("✓ Import FidelityQuantumKernel jako QuantumKernel - sukces")
        except ImportError:
            print("❌ Błąd: Nie można zaimportować QuantumKernel")
            print("Sprawdź wersję qiskit-machine-learning: pip install --upgrade qiskit-machine-learning")
            sys.exit(1)

# Naprawiony import QSVC
try:
    from qiskit_machine_learning.algorithms import QSVC
    print("✓ Import QSVC z qiskit_machine_learning.algorithms - sukces")
except ImportError:
    try:
        from qiskit_machine_learning.algorithms.svm import QSVC
        print("✓ Import QSVC z qiskit_machine_learning.algorithms.svm - sukces")
    except ImportError:
        print("❌ Błąd: Nie można zaimportować QSVC")
        print("Sprawdź wersję qiskit-machine-learning: pip install --upgrade qiskit-machine-learning")
        sys.exit(1)
# import dimod  # Nie używane - usunięte
# import neal    # Nie używane - usunięte

# ----------------- PARAMETRY KONFIGURACYJNE -----------------

# Wybór zmiennej docelowej
TARGET_VARIABLE = 'Primary_Diagnosis'  # Możliwe wartości: 'Primary_Diagnosis' lub 'Grade'

# Wybór map cech do aktywacji w eksperymentach - UPROSZCZONE
FEATURE_MAPS_ENABLED = {
    'ZZ1': False,            # Tylko proste mapy cech
    'ZZ2': False,           # Wyłączone - zbyt złożone
    'Pauli1': True,         # Tylko 1 powtórzenie
    'Pauli2': False,        # Wyłączone - zbyt złożone
    'Z1': True,             # Prosta implementacja Z-FeatureMap
    'Z2': True,            # Wyłączone - zbyt złożone
    'ZZ1_new': False,       # Wyłączone
    'ZZ2_new': False,       # Wyłączone
    'SU2_1': False,         # Wyłączone - zbyt złożone
    'SU2_2': False,         # Wyłączone
    'SU2_full': False,      # Wyłączone
    'Amplitude_l2': False,   # Kodowanie amplitudowe - szybsze
    'Amplitude_l1': False,  # Wyłączone dla uproszczenia
    'Amplitude_min-max': False  # Wyłączone dla uproszczenia
}

# Parametry danych
DATA_FILE = 'dane/TCGA_GBM_LGG_Mutations_clean.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42
PCA_COMPONENTS = 12  # Zmniejszone z 16 na 8 dla szybszych obliczeń

# Parametry bezpieczeństwa i wydajności
QUANTUM_SHOTS = 50  # Zmniejszone z 1024 na 50
QUANTUM_TIMEOUT = 3000  # 5 minut timeout dla operacji kwantowych
MAX_FEATURE_DIMENSION = 8  # Maksymalny wymiar dla map cech
USE_PARALLEL = False  # Wyłączone równoległe przetwarzanie
CHECKPOINT_INTERVAL = 5  # Zapisuj wyniki co 5 iteracji

# Wybór eksperymentów do przeprowadzenia
RUN_COMPLEXITY_EXPERIMENT = False  # Wyłączone - zbyt czasochłonne
RUN_GENE_SUBSETS_EXPERIMENT = True
RUN_FEATURE_MAPPINGS_EXPERIMENT = False  # Wyłączone - problematyczne

# Parametry wyjściowe
OUTPUT_DIR = 'wyniki/wyniki_eksperymentow_1109-2'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'wyniki_eksperymentow_{TARGET_VARIABLE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Upewnij się, że katalog wyjściowy istnieje
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Klasa do przekierowania wyjścia do pliku i terminala jednocześnie
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.checkpoint_count = 0

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
        # Checkpoint co pewien czas
        if 'Testing' in message and 'for' in message:
            self.checkpoint_count += 1
            if self.checkpoint_count % CHECKPOINT_INTERVAL == 0:
                self.write(f'\n*** CHECKPOINT {self.checkpoint_count} - {datetime.now().strftime("%H:%M:%S")} ***\n')

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Funkcja do bezpiecznego tworzenia backend'u kwantowego
def get_quantum_backend():
    """
    Tworzy backend kwantowy z obsługą różnych wersji Qiskit.
    
    Returns:
        Backend kwantowy
    """
    try:
        # Próbuj różne sposoby importu Aer
        try:
            from qiskit import Aer
            return Aer.get_backend('qasm_simulator')
        except ImportError:
            try:
                from qiskit_aer import Aer
                return Aer.get_backend('qasm_simulator')
            except ImportError:
                from qiskit.providers.aer import Aer
                return Aer.get_backend('qasm_simulator')
    except Exception as e:
        print(f"Błąd podczas tworzenia backend'u: {e}")
        # Fallback - użyj podstawowego backend'u
        try:
            from qiskit import BasicAer
            return BasicAer.get_backend('qasm_simulator')
        except:
            raise RuntimeError("Nie można utworzyć backend'u kwantowego")

def create_quantum_kernel(feature_map, backend_or_instance=None):
    """
    Tworzy jądro kwantowe z obsługą różnych wersji Qiskit Machine Learning.
    
    Args:
        feature_map: Mapa cech kwantowych
        backend_or_instance: Backend kwantowy lub quantum_instance (opcjonalny)
        
    Returns:
        Jądro kwantowe
    """
    if backend_or_instance is None:
        backend_or_instance = get_quantum_backend()
    
    try:
        # Próbuj różne sposoby tworzenia jądra kwantowego
        try:
            # Nowa wersja - FidelityQuantumKernel (tylko quantum_instance)
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            if hasattr(backend_or_instance, 'backend'):  # quantum_instance
                return FidelityQuantumKernel(feature_map=feature_map, quantum_instance=backend_or_instance)
            else:  # backend - musimy utworzyć quantum_instance
                from qiskit.utils import QuantumInstance
                quantum_instance = QuantumInstance(backend=backend_or_instance, shots=QUANTUM_SHOTS)
                return FidelityQuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
        except ImportError:
            try:
                # Stara wersja - QuantumKernel
                if hasattr(backend_or_instance, 'backend'):  # quantum_instance
                    return QuantumKernel(feature_map=feature_map, quantum_instance=backend_or_instance)
                else:  # backend
                    return QuantumKernel(feature_map=feature_map, backend=backend_or_instance)
            except TypeError:
                # Jeszcze starsza wersja - bez parametru backend
                return QuantumKernel(feature_map=feature_map)
    except Exception as e:
        print(f"Błąd podczas tworzenia jądra kwantowego: {e}")
        raise RuntimeError("Nie można utworzyć jądra kwantowego")

# Funkcja timeout dla operacji kwantowych
def timeout_handler(signum, frame):
    raise TimeoutError("Operacja kwantowa przekroczyła limit czasu")

# Funkcja bezpiecznego wykonywania operacji kwantowych
def safe_quantum_operation(operation, timeout=QUANTUM_TIMEOUT):
    """
    Wykonuje operację kwantową z timeout.
    
    Args:
        operation: Funkcja do wykonania
        timeout: Maksymalny czas w sekundach
        
    Returns:
        Wynik operacji lub None w przypadku błędu
    """
    try:
        # Ustaw timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Wykonaj operację
        result = operation()
        
        # Wyłącz timeout
        signal.alarm(0)
        return result
        
    except TimeoutError:
        print(f"    Timeout po {timeout} sekundach - pomijam")
        return None
    except Exception as e:
        print(f"    Błąd: {str(e)}")
        return None
    finally:
        signal.alarm(0)

# ----------------- FUNKCJE POMOCNICZE -----------------

def z_feature_map(feature_dimension, reps=1):
    """
    Tworzy niestandardową mapę cech Z-FeatureMap.
    
    Args:
        feature_dimension: Wymiar wektora cech
        reps: Liczba powtórzeń
        
    Returns:
        QuantumCircuit: Układ kwantowy reprezentujący mapę cech
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    
    # Tworzymy układ kwantowy z odpowiednią liczbą kubitów
    qc = QuantumCircuit(feature_dimension)
    
    # Tworzymy wektor parametrów
    params = ParameterVector('x', feature_dimension)
    
    # Dodajemy bramki Hadamarda na wszystkich kubitach
    for i in range(feature_dimension):
        qc.h(i)
    
    # Powtarzamy blok reps razy
    for _ in range(reps):
        # Dodajemy rotacje Z na każdym kubicie
        for i in range(feature_dimension):
            qc.rz(params[i], i)
    
    return qc

def zz_feature_map(feature_dimension, reps=1):
    """
    Tworzy niestandardową mapę cech ZZ-FeatureMap.
    
    Args:
        feature_dimension: Wymiar wektora cech
        reps: Liczba powtórzeń
        
    Returns:
        QuantumCircuit: Układ kwantowy reprezentujący mapę cech
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    
    # Tworzymy układ kwantowy z odpowiednią liczbą kubitów
    qc = QuantumCircuit(feature_dimension)
    
    # Tworzymy wektor parametrów
    params = ParameterVector('x', feature_dimension)
    
    # Dodajemy bramki Hadamarda na wszystkich kubitach
    for i in range(feature_dimension):
        qc.h(i)
    
    # Powtarzamy blok reps razy
    for _ in range(reps):
        # Dodajemy rotacje Z na każdym kubicie
        for i in range(feature_dimension):
            qc.rz(params[i], i)
        
        # Dodajemy bramki ZZ między sąsiednimi kubitami
        for i in range(feature_dimension - 1):
            qc.cx(i, i+1)
            qc.rz(params[i] * params[i+1], i+1)
            qc.cx(i, i+1)
    
    return qc

# Funkcje eksperymentalne

def experiment_complexity_impact(X, y, mutation_counts):
    """
    Bada wpływ złożoności danych na skuteczność klasyfikacji.
    
    Parameters:
    -----------
    X : DataFrame
        Dane wejściowe (cechy)
    y : Series
        Etykiety klas (zmienna docelowa określona przez TARGET_VARIABLE)
    mutation_counts : Series
        Liczba mutacji dla każdego przypadku
    
    Returns:
    --------
    dict
        Słownik z wynikami dla każdego poziomu złożoności
    """
    # Definiuj poziomy złożoności na podstawie liczby mutacji
    complexity_levels = {}
    
    # Poziom 1: Przypadki z małą liczbą mutacji (dolny kwartyl)
    low_threshold = mutation_counts.quantile(0.25)
    complexity_levels['Low Complexity'] = mutation_counts[mutation_counts <= low_threshold].index.tolist()
    
    # Poziom 2: Przypadki ze średnią liczbą mutacji (między dolnym a górnym kwartylem)
    high_threshold = mutation_counts.quantile(0.75)
    complexity_levels['Medium Complexity'] = mutation_counts[(mutation_counts > low_threshold) & 
                                                           (mutation_counts < high_threshold)].index.tolist()
    
    # Poziom 3: Przypadki z dużą liczbą mutacji (górny kwartyl)
    complexity_levels['High Complexity'] = mutation_counts[mutation_counts >= high_threshold].index.tolist()
    
    # Wyświetl informacje o poziomach złożoności
    print("\nPoziomy złożoności danych:")
    for level, indices in complexity_levels.items():
        print(f"  {level}: {len(indices)} przypadków")
        avg_mutations = mutation_counts.iloc[indices].mean()
        print(f"    Średnia liczba mutacji: {avg_mutations:.2f}")
    
    results = {}
    
    # Przygotuj wszystkie możliwe mapowania cech - podobnie jak w experiment_feature_mappings
    def prepare_feature_maps(feature_dimension):
        """
        Przygotowuje wszystkie możliwe mapowania cech dla danego wymiaru.
        
        Args:
            feature_dimension: Wymiar wektora cech
            
        Returns:
            Lista słowników z konfiguracjami map cech
        """
        all_feature_maps = [
            {'name': 'ZZ1_C10', 'base_name': 'ZZ1', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=1), 'C': 10.0},
            {'name': 'ZZ2_C01', 'base_name': 'ZZ2', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=2), 'C': 0.1},
            {'name': 'Pauli1_C10', 'base_name': 'Pauli1', 'map': PauliFeatureMap(feature_dimension=feature_dimension, reps=1), 'C': 10.0},
            {'name': 'ZZ1_C1', 'base_name': 'ZZ1', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=1), 'C': 1.0},
            {'name': 'ZZ2_C1', 'base_name': 'ZZ2', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=2), 'C': 1.0},
            {'name': 'Pauli1_C1', 'base_name': 'Pauli1', 'map': PauliFeatureMap(feature_dimension=feature_dimension, reps=1), 'C': 1.0},
            {'name': 'Pauli2_C1', 'base_name': 'Pauli2', 'map': PauliFeatureMap(feature_dimension=feature_dimension, reps=2), 'C': 1.0},
            # Dodaj nowe mapy cech Z-FeatureMap
            {'name': 'Z1_C01', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=feature_dimension, reps=1), 'C': 0.1},
            {'name': 'Z1_C1', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=feature_dimension, reps=1), 'C': 1.0},
            {'name': 'Z1_C10', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=feature_dimension, reps=1), 'C': 10.0},
            {'name': 'Z2_C1', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=feature_dimension, reps=2), 'C': 1.0},
            {'name': 'Z2_C01', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=feature_dimension, reps=2), 'C': 0.1},
            {'name': 'Z2_C10', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=feature_dimension, reps=2), 'C': 10.0},
            # Dodaj nowe mapy cech ZZ-FeatureMap (nowa implementacja)
            {'name': 'ZZ1_new_C1', 'base_name': 'ZZ1_new', 'map': zz_feature_map(feature_dimension=feature_dimension, reps=1), 'C': 1.0},
            {'name': 'ZZ2_new_C1', 'base_name': 'ZZ2_new', 'map': zz_feature_map(feature_dimension=feature_dimension, reps=2), 'C': 1.0},
            # Dodaj mapy cech SU2
            {'name': 'SU2_1_C1', 'base_name': 'SU2_1', 'map': EfficientSU2(num_qubits=feature_dimension, entanglement='linear', reps=1), 'C': 1.0},
            {'name': 'SU2_2_C1', 'base_name': 'SU2_2', 'map': EfficientSU2(num_qubits=feature_dimension, entanglement='linear', reps=2), 'C': 1.0},
            {'name': 'SU2_full_C1', 'base_name': 'SU2_full', 'map': EfficientSU2(num_qubits=feature_dimension, entanglement='full', reps=1), 'C': 1.0},
            # Dodaj oficjalną implementację ZFeatureMap
            {'name': 'ZFeatureMap1_C1', 'base_name': 'ZFeatureMap1', 'map': ZFeatureMap(feature_dimension=feature_dimension, reps=1), 'C': 1.0},
            {'name': 'ZFeatureMap2_C1', 'base_name': 'ZFeatureMap2', 'map': ZFeatureMap(feature_dimension=feature_dimension, reps=2), 'C': 1.0},
            # Dodaj kodowanie amplitudowe
            {'name': 'Amplitude_l2_C1', 'base_name': 'Amplitude_l2', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'l2'},
            {'name': 'Amplitude_l1_C1', 'base_name': 'Amplitude_l1', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'l1'},
            {'name': 'Amplitude_min-max_C1', 'base_name': 'Amplitude_min-max', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'min-max'}
        ]
        
        # Filtruj mapy cech na podstawie ustawień FEATURE_MAPS_ENABLED
        feature_maps = []
        for fm in all_feature_maps:
            if fm['base_name'] in FEATURE_MAPS_ENABLED and FEATURE_MAPS_ENABLED[fm['base_name']]:
                feature_maps.append(fm)
        
        return feature_maps
    
    # Dla każdego poziomu złożoności
    for complexity_level, indices in complexity_levels.items():
        print(f"\nTestowanie poziomu złożoności: {complexity_level}")
        
        # Wybierz podzbiór danych
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]
        
        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Skalowanie danych
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Redukcja wymiarowości dla modelu kwantowego
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)
        
        # Klasyczny SVM (liniowy)
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(X_train_scaled, y_train)
        linear_acc = svm_linear.score(X_test_scaled, y_test)
        
        # Klasyczny SVM (RBF)
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(X_train_scaled, y_train)
        rbf_acc = svm_rbf.score(X_test_scaled, y_test)
        
        # Przygotuj wyniki dla kwantowych SVM
        quantum_results = {}
        
        # Przygotuj wszystkie mapy cech
        feature_maps = prepare_feature_maps(X_train_reduced.shape[1])
        print(f"Aktywne mapy cech: {[fm['name'] for fm in feature_maps]}")
        
        # Normalizacja danych do zakresu [0, 2π]
        X_train_normalized = X_train_reduced.copy()
        X_test_normalized = X_test_reduced.copy()
        
        for i in range(X_train_reduced.shape[1]):
            min_val = min(X_train_reduced[:, i].min(), X_test_reduced[:, i].min())
            max_val = max(X_train_reduced[:, i].max(), X_test_reduced[:, i].max())
            
            if max_val > min_val:
                X_train_normalized[:, i] = 2 * np.pi * (X_train_reduced[:, i] - min_val) / (max_val - min_val)
                X_test_normalized[:, i] = 2 * np.pi * (X_test_reduced[:, i] - min_val) / (max_val - min_val)
        
        # Testuj każdą mapę cech - SEKWENCYJNIE z timeout
        for i, fm in enumerate(feature_maps):
            print(f"  Testowanie mapy cech: {fm['name']} ({i+1}/{len(feature_maps)})")
            
            def train_quantum_model():
                try:
                    # Używamy quantum_instance zamiast backend
                    quantum_instance = qiskit.utils.QuantumInstance(
                        backend=get_quantum_backend(),
                        shots=QUANTUM_SHOTS
                    )
                    quantum_kernel = create_quantum_kernel(fm['map'], quantum_instance)
                except (TypeError, AttributeError):
                    # Dla starszych wersji Qiskit
                    backend = get_quantum_backend()
                    backend.shots = QUANTUM_SHOTS
                    quantum_kernel = create_quantum_kernel(fm['map'], backend)
                
                # Używamy quantum_kernel bezpośrednio z QSVC
                qsvm = QSVC(quantum_kernel=quantum_kernel, C=fm['C'])
                
                # Trenuj model
                qsvm.fit(X_train_normalized, y_train)
                
                # Testuj model
                quantum_acc = qsvm.score(X_test_normalized, y_test)
                return quantum_acc
            
            # Wykonaj z timeout
            result = safe_quantum_operation(train_quantum_model, QUANTUM_TIMEOUT)
            
            if result is not None:
                quantum_results[fm['name']] = result
                print(f"    Dokładność: {result:.4f}")
            else:
                quantum_results[fm['name']] = 0.0
                print(f"    Pominięte z powodu błędu/timeout")
        
        # Znajdź najlepszy model kwantowy
        best_quantum_acc = max(quantum_results.values()) if quantum_results else 0.0
        best_quantum_model = max(quantum_results.items(), key=lambda x: x[1])[0] if quantum_results else "None"
        
        results[complexity_level] = {
            'linear_svm': linear_acc,
            'rbf_svm': rbf_acc,
            'best_quantum_svm': best_quantum_acc,
            'best_quantum_model': best_quantum_model
        }
        
        # Dodaj wyniki dla każdej mapy cech
        for name, acc in quantum_results.items():
            results[complexity_level][name] = acc
        
        print(f"  Linear SVM: {linear_acc:.4f}")
        print(f"  RBF SVM: {rbf_acc:.4f}")
        print(f"  Best Quantum SVM: {best_quantum_acc:.4f} ({best_quantum_model})")
    
    return results

def experiment_gene_subsets(data):
    """
    Bada wpływ różnych podzbiorów genów na skuteczność klasyfikacji.
    
    Parameters:
    -----------
    data : DataFrame
        Pełny zbiór danych
    
    Returns:
    --------
    dict
        Słownik z wynikami dla każdego podzbioru genów
    """
    # Definiuj podzbiory genów
    gene_subsets = {
        "Wszystkie geny": None,  # Wszystkie geny
        "Geny często mutowane": ["IDH1", "TP53", "ATRX", "PTEN"],
        "Geny średnio mutowane": ["EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1"],
        "Geny rzadko mutowane": ["RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4"],
    }
    
    # Oblicz liczbę genów w każdym podzbiorze i liczbę dostępnych genów w danych
    gene_counts = {}
    for name, genes in gene_subsets.items():
        if genes:
            available_genes = [g for g in genes if g in data.columns]
            gene_counts[name] = {
                'total': len(genes),
                'available': len(available_genes),
                'missing': len(genes) - len(available_genes)
            }
        else:
            # Dla "All Genes" policz wszystkie kolumny genów
            gene_cols = [col for col in data.columns if col not in 
                        ['Grade', 'Project', 'Case_ID', 'Gender', 'Age_at_diagnosis', 
                         'Primary_Diagnosis', 'Race']]
            gene_counts[name] = {
                'total': len(gene_cols),
                'available': len(gene_cols),
                'missing': 0
            }
    
    print("\nLiczba genów w każdym podzbiorze:")
    for name, counts in gene_counts.items():
        print(f"  {name}: {counts['total']} genów (dostępne: {counts['available']}, brakujące: {counts['missing']})")
    
    results = {}
    
    # Dla każdego podzbioru genów
    for name, genes in gene_subsets.items():
        print(f"\nTesting gene subset: {name} (n={len(genes) if genes else 'all'})")
        
        # Przygotuj dane
        if genes:
            # Sprawdź, które geny są dostępne w danych
            available_genes = [g for g in genes if g in data.columns]
            if len(available_genes) < len(genes):
                print(f"  Warning: Only {len(available_genes)}/{len(genes)} genes are available in the dataset")
            
            if not available_genes:
                print("  Skipping due to no available genes")
                continue
                
            # Wybierz tylko określone geny
            X = data[available_genes].copy()
            sample_size = len(genes)
        else:
            # Użyj wszystkich genów
            gene_cols = [col for col in data.columns if col not in 
                        ['Grade', 'Project', 'Case_ID', 'Gender', 'Age_at_diagnosis', 
                         'Primary_Diagnosis', 'Race']]
            X = data[gene_cols].copy()
            sample_size = len(gene_cols)  # Liczba wszystkich genów
        
        # Konwertuj dane tekstowe na binarne (0/1)
        for col in X.columns:
            if X[col].dtype == 'object':  # Jeśli kolumna zawiera dane tekstowe
                X[col] = X[col].apply(lambda x: 1 if str(x).upper() == 'MUTATED' else 0)
        
        # Użyj zmiennej docelowej określonej w konfiguracji
        y = data[TARGET_VARIABLE]
        
        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Skalowanie danych
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Redukcja wymiarowości dla modelu kwantowego - ograniczona do MAX_FEATURE_DIMENSION
        max_components = min(PCA_COMPONENTS, MAX_FEATURE_DIMENSION, X_train_scaled.shape[1])
        if X_train_scaled.shape[1] > max_components:
            pca = PCA(n_components=max_components)
            X_train_reduced = pca.fit_transform(X_train_scaled)
            X_test_reduced = pca.transform(X_test_scaled)
        else:
            X_train_reduced = X_train_scaled
            X_test_reduced = X_test_scaled
        
        print(f"    Redukcja wymiarowości: {X_train_scaled.shape[1]} -> {max_components} wymiarów")
        
        # Klasyczny SVM (liniowy)
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(X_train_scaled, y_train)
        linear_acc = svm_linear.score(X_test_scaled, y_test)
        
        # Klasyczny SVM (RBF)
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(X_train_scaled, y_train)
        rbf_acc = svm_rbf.score(X_test_scaled, y_test)
        
        # Przygotuj wyniki dla kwantowych SVM
        quantum_results = {}
        
        # Konfiguracja 1: ZZ1, C=10.0 - z timeout
        print("  Testing ZZ1, C=10.0")
        
        def train_zz1_model():
            feature_map_zz1 = ZZFeatureMap(feature_dimension=X_train_reduced.shape[1], reps=1)
            try:
                # Używamy quantum_instance zamiast backend
                quantum_instance = qiskit.utils.QuantumInstance(
                    backend=get_quantum_backend(),
                    shots=QUANTUM_SHOTS
                )
                quantum_kernel_zz1 = create_quantum_kernel(feature_map_zz1, quantum_instance)
            except (TypeError, AttributeError):
                # Dla starszych wersji Qiskit
                backend = get_quantum_backend()
                backend.shots = QUANTUM_SHOTS
                quantum_kernel_zz1 = create_quantum_kernel(feature_map_zz1, backend)
            
            # Używamy quantum_kernel bezpośrednio z QSVC
            qsvm_zz1 = QSVC(quantum_kernel=quantum_kernel_zz1, C=10.0)
            
            # Trenuj model
            qsvm_zz1.fit(X_train_reduced, y_train)
            
            # Testuj model
            zz1_acc = qsvm_zz1.score(X_test_reduced, y_test)
            return zz1_acc
        
        result = safe_quantum_operation(train_zz1_model, QUANTUM_TIMEOUT)
        if result is not None:
            quantum_results['ZZ1_C10'] = result
            print(f"    Accuracy: {result:.4f}")
        else:
            quantum_results['ZZ1_C10'] = 0.0
            print(f"    Pominięte z powodu błędu/timeout")
        
        # Konfiguracja 2: ZZ2, C=0.1 - z timeout
        print("  Testing ZZ2, C=0.1")
        
        def train_zz2_model():
            feature_map_zz2 = ZZFeatureMap(feature_dimension=X_train_reduced.shape[1], reps=2)
            try:
                # Używamy quantum_instance zamiast backend
                quantum_instance = qiskit.utils.QuantumInstance(
                    backend=get_quantum_backend(),
                    shots=QUANTUM_SHOTS
                )
                quantum_kernel_zz2 = create_quantum_kernel(feature_map_zz2, quantum_instance)
            except (TypeError, AttributeError):
                # Dla starszych wersji Qiskit
                backend = get_quantum_backend()
                backend.shots = QUANTUM_SHOTS
                quantum_kernel_zz2 = create_quantum_kernel(feature_map_zz2, backend)
            
            # Używamy quantum_kernel bezpośrednio z QSVC
            qsvm_zz2 = QSVC(quantum_kernel=quantum_kernel_zz2, C=0.1)
            
            # Trenuj model
            qsvm_zz2.fit(X_train_reduced, y_train)
            
            # Testuj model
            zz2_acc = qsvm_zz2.score(X_test_reduced, y_test)
            return zz2_acc
        
        result = safe_quantum_operation(train_zz2_model, QUANTUM_TIMEOUT)
        if result is not None:
            quantum_results['ZZ2_C01'] = result
            print(f"    Accuracy: {result:.4f}")
        else:
            quantum_results['ZZ2_C01'] = 0.0
            print(f"    Pominięte z powodu błędu/timeout")
        
        # Konfiguracja 3: Pauli1, C=10.0 - z timeout
        print("  Testing Pauli1, C=10.0")
        
        def train_pauli1_model():
            feature_map_pauli1 = PauliFeatureMap(feature_dimension=X_train_reduced.shape[1], reps=1)
            try:
                # Używamy quantum_instance zamiast backend
                quantum_instance = qiskit.utils.QuantumInstance(
                    backend=get_quantum_backend(),
                    shots=QUANTUM_SHOTS
                )
                quantum_kernel_pauli1 = create_quantum_kernel(feature_map_pauli1, quantum_instance)
            except (TypeError, AttributeError):
                # Dla starszych wersji Qiskit
                backend = get_quantum_backend()
                backend.shots = QUANTUM_SHOTS
                quantum_kernel_pauli1 = create_quantum_kernel(feature_map_pauli1, backend)
            
            # Używamy quantum_kernel bezpośrednio z QSVC
            qsvm_pauli1 = QSVC(quantum_kernel=quantum_kernel_pauli1, C=10.0)
            
            # Trenuj model
            qsvm_pauli1.fit(X_train_reduced, y_train)
            
            # Testuj model
            pauli1_acc = qsvm_pauli1.score(X_test_reduced, y_test)
            return pauli1_acc
        
        result = safe_quantum_operation(train_pauli1_model, QUANTUM_TIMEOUT)
        if result is not None:
            quantum_results['Pauli1_C10'] = result
            print(f"    Accuracy: {result:.4f}")
        else:
            quantum_results['Pauli1_C10'] = 0.0
            print(f"    Pominięte z powodu błędu/timeout")
        
        # Znajdź najlepszy model kwantowy
        best_quantum_acc = max(quantum_results.values()) if quantum_results else 0.0
        best_quantum_model = max(quantum_results.items(), key=lambda x: x[1])[0] if quantum_results else "None"
        
        results[name] = {
            'linear_svm': linear_acc,
            'rbf_svm': rbf_acc,
            'best_quantum_svm': best_quantum_acc,
            'best_quantum_model': best_quantum_model,
            'ZZ1_C10': quantum_results.get('ZZ1_C10', 0.0),
            'ZZ2_C01': quantum_results.get('ZZ2_C01', 0.0),
            'Pauli1_C10': quantum_results.get('Pauli1_C10', 0.0),
            'sample_size': sample_size
        }
        
        print(f"  Linear SVM: {linear_acc:.4f}")
        print(f"  RBF SVM: {rbf_acc:.4f}")
        print(f"  Best Quantum SVM: {best_quantum_acc:.4f} ({best_quantum_model})")
        
    return results

def experiment_feature_mappings(X_train, X_test, y_train, y_test):
    """
    Testuje różne mapowania cech kwantowych i analizuje ich wpływ na wydajność.
    
    Parameters:
    -----------
    X_train, X_test : array
        Dane treningowe i testowe po redukcji wymiarowości
    y_train, y_test : array
        Etykiety treningowe i testowe
    
    Returns:
    --------
    dict
        Słownik z wynikami dla każdego mapowania cech
    """
    # Normalizacja danych do zakresu [0, 2π]
    X_train_normalized = X_train.copy()
    X_test_normalized = X_test.copy()
    
    for i in range(X_train.shape[1]):
        min_val = min(X_train[:, i].min(), X_test[:, i].min())
        max_val = max(X_train[:, i].max(), X_test[:, i].max())
        
        if max_val > min_val:
            X_train_normalized[:, i] = 2 * np.pi * (X_train[:, i] - min_val) / (max_val - min_val)
            X_test_normalized[:, i] = 2 * np.pi * (X_test[:, i] - min_val) / (max_val - min_val)
    
    # Funkcja do tworzenia Z-FeatureMap
    def z_feature_map(feature_dimension, reps=1):
        """
        Tworzy Z-FeatureMap - prostą mapę cech opartą na bramkach Z.
        
        Parameters:
        -----------
        feature_dimension : int
            Liczba wymiarów wektora cech (liczba kubitów)
        reps : int
            Liczba powtórzeń obwodu
            
        Returns:
        --------
        QuantumCircuit
            Obwód kwantowy implementujący Z-FeatureMap
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter, ParameterVector
        
        qc = QuantumCircuit(feature_dimension)
        
        # Parametry dla każdego kubitu
        params = ParameterVector(f'x', feature_dimension)
        
        # Powtórz obwód określoną liczbę razy
        for _ in range(reps):
            # Hadamard na wszystkich kubitach
            for i in range(feature_dimension):
                qc.h(i)
            
            # Rotacje Z na każdym kubicie
            for i in range(feature_dimension):
                qc.rz(params[i], i)
        
        return qc
    
    # Przygotuj wszystkie możliwe mapowania cech
    all_feature_maps = [
        {'name': 'ZZ1_C10', 'base_name': 'ZZ1', 'map': ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1), 'C': 10.0},
        {'name': 'ZZ2_C01', 'base_name': 'ZZ2', 'map': ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2), 'C': 0.1},
        {'name': 'Pauli1_C10', 'base_name': 'Pauli1', 'map': PauliFeatureMap(feature_dimension=X_train.shape[1], reps=1), 'C': 10.0},
        {'name': 'ZZ1_C1', 'base_name': 'ZZ1', 'map': ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1), 'C': 1.0},
        {'name': 'ZZ2_C1', 'base_name': 'ZZ2', 'map': ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2), 'C': 1.0},
        {'name': 'Pauli1_C1', 'base_name': 'Pauli1', 'map': PauliFeatureMap(feature_dimension=X_train.shape[1], reps=1), 'C': 1.0},
        {'name': 'Pauli2_C1', 'base_name': 'Pauli2', 'map': PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2), 'C': 1.0},
        # Dodaj nowe mapy cech Z-FeatureMap
        {'name': 'Z1_C01', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=1), 'C': 0.1},
        {'name': 'Z1_C1', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=1), 'C': 1.0},
        {'name': 'Z1_C10', 'base_name': 'Z1', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=1), 'C': 10.0},
        {'name': 'Z2_C1', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=2), 'C': 1.0},
        {'name': 'Z2_C01', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=2), 'C': 0.1},
        {'name': 'Z2_C10', 'base_name': 'Z2', 'map': z_feature_map(feature_dimension=X_train.shape[1], reps=2), 'C': 10.0},
        # Dodaj nowe mapy cech ZZ-FeatureMap (nowa implementacja)
        {'name': 'ZZ1_new_C1', 'base_name': 'ZZ1_new', 'map': zz_feature_map(feature_dimension=X_train.shape[1], reps=1), 'C': 1.0},
        {'name': 'ZZ2_new_C1', 'base_name': 'ZZ2_new', 'map': zz_feature_map(feature_dimension=X_train.shape[1], reps=2), 'C': 1.0},
        # Dodaj mapy cech SU2
        {'name': 'SU2_1_C1', 'base_name': 'SU2_1', 'map': EfficientSU2(num_qubits=X_train.shape[1], entanglement='linear', reps=1), 'C': 1.0},
        {'name': 'SU2_2_C1', 'base_name': 'SU2_2', 'map': EfficientSU2(num_qubits=X_train.shape[1], entanglement='linear', reps=2), 'C': 1.0},
        {'name': 'SU2_full_C1', 'base_name': 'SU2_full', 'map': EfficientSU2(num_qubits=X_train.shape[1], entanglement='full', reps=1), 'C': 1.0},
        # Dodaj oficjalną implementację ZFeatureMap
        {'name': 'ZFeatureMap1_C1', 'base_name': 'ZFeatureMap1', 'map': ZFeatureMap(feature_dimension=X_train.shape[1], reps=1), 'C': 1.0},
        {'name': 'ZFeatureMap2_C1', 'base_name': 'ZFeatureMap2', 'map': ZFeatureMap(feature_dimension=X_train.shape[1], reps=2), 'C': 1.0},
        # Dodaj kodowanie amplitudowe
        {'name': 'Amplitude_l2_C1', 'base_name': 'Amplitude_l2', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'l2'},
        {'name': 'Amplitude_l1_C1', 'base_name': 'Amplitude_l1', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'l1'},
        {'name': 'Amplitude_min-max_C1', 'base_name': 'Amplitude_min-max', 'map': None, 'C': 1.0, 'amplitude_encoding': True, 'normalization': 'min-max'}
    ]
    
    # Filtruj mapy cech na podstawie ustawień FEATURE_MAPS_ENABLED
    feature_maps = []
    for fm in all_feature_maps:
        if fm['base_name'] in FEATURE_MAPS_ENABLED and FEATURE_MAPS_ENABLED[fm['base_name']]:
            feature_maps.append(fm)
    
    print(f"Aktywne mapy cech: {[fm['name'] for fm in feature_maps]}")
    
    results = {}
    
    # Dla porównania, trenuj klasyczny SVM z jądrem RBF
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    rbf_acc = svm_rbf.score(X_test, y_test)
    print(f"Dokładność klasycznego SVM (RBF): {rbf_acc:.4f}")
    
    # Testuj każde mapowanie cech - SEKWENCYJNIE z timeout
    for i, fm in enumerate(feature_maps):
        print(f"\nTestowanie mapowania: {fm['name']} ({i+1}/{len(feature_maps)})")
        
        def train_feature_mapping_model():
            # Utwórz jądro kwantowe
            try:
                # Używamy quantum_instance zamiast backend
                quantum_instance = qiskit.utils.QuantumInstance(
                    backend=get_quantum_backend(),
                    shots=QUANTUM_SHOTS
                )
                quantum_kernel = create_quantum_kernel(fm['map'], quantum_instance)
            except (TypeError, AttributeError):
                # Dla starszych wersji Qiskit
                backend = get_quantum_backend()
                backend.shots = QUANTUM_SHOTS
                quantum_kernel = create_quantum_kernel(fm['map'], backend)
            
            # Używamy quantum_kernel bezpośrednio z QSVC
            qsvm = QSVC(quantum_kernel=quantum_kernel, C=fm['C'])
            
            # Trenuj model
            qsvm.fit(X_train_normalized, y_train)
            
            # Testuj model
            quantum_acc = qsvm.score(X_test_normalized, y_test)
            
            # Oblicz macierz jądra na niewielkiej próbce
            sample_size = min(20, len(X_train_normalized))  # Zmniejszone z 50 na 20
            sample_indices = np.random.choice(len(X_train_normalized), sample_size, replace=False)
            
            try:
                kernel_matrix = quantum_kernel.evaluate(
                    X_train_normalized[sample_indices], 
                    X_train_normalized[sample_indices]
                )
                # Weź tylko część rzeczywistą macierzy jądra dla analizy
                kernel_matrix = np.real(kernel_matrix)
            except Exception as e:
                print(f"  Błąd podczas obliczania macierzy jądra: {str(e)}")
                kernel_matrix = np.eye(sample_size)  # Macierz jednostkowa jako fallback
            
            # Analizuj właściwości macierzy jądra
            kernel_properties = {
                'mean': np.mean(kernel_matrix),
                'std': np.std(kernel_matrix),
                'min': np.min(kernel_matrix),
                'max': np.max(kernel_matrix)
            }
            
            return quantum_acc, kernel_properties
        
        # Wykonaj z timeout
        result = safe_quantum_operation(train_feature_mapping_model, QUANTUM_TIMEOUT)
        
        if result is not None:
            quantum_acc, kernel_properties = result
            results[fm['name']] = {
                'accuracy': quantum_acc,
                'kernel_properties': kernel_properties,
                'C': fm['C']
            }
            
            print(f"  Dokładność: {quantum_acc:.4f}")
            print(f"  Właściwości macierzy jądra: śr={kernel_properties['mean']:.3f}, "
                  f"std={kernel_properties['std']:.3f}, min={kernel_properties['min']:.3f}, "
                  f"max={kernel_properties['max']:.3f}")
        else:
            results[fm['name']] = {
                'accuracy': 0.0,
                'kernel_properties': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                'C': fm['C']
            }
            print(f"  Pominięte z powodu błędu/timeout")
    
    # Porównaj z klasycznym SVM
    quantum_accuracies = [results[fm['name']]['accuracy'] for fm in feature_maps]
    if quantum_accuracies:
        best_quantum_acc = max(quantum_accuracies)
        best_map = feature_maps[np.argmax(quantum_accuracies)]['name']
        
        print(f"\nNajlepsze mapowanie kwantowe: {best_map} (dokładność: {best_quantum_acc:.4f})")
        print(f"Różnica względem klasycznego SVM: {best_quantum_acc - rbf_acc:.4f}")
    else:
        print("\nNie udało się wytrenować żadnego modelu kwantowego.")
    
    return results

# Dodaj funkcje do kodowania amplitudowego
def prepare_data_for_amplitude_encoding(X, normalization='l2'):
    """
    Przygotowuje dane do kodowania amplitudowego.
    
    Args:
        X: Dane wejściowe
        normalization: Typ normalizacji ('l2', 'l1', lub 'min-max')
        
    Returns:
        Przygotowane dane
    """
    X_prepared = X.copy()
    
    # Normalizacja danych
    if normalization == 'l2':
        # Normalizacja L2 (suma kwadratów = 1)
        norms = np.linalg.norm(X_prepared, axis=1, ord=2)
        # Unikaj dzielenia przez zero
        norms[norms == 0] = 1.0
        X_prepared = X_prepared / norms[:, np.newaxis]
    elif normalization == 'l1':
        # Normalizacja L1 (suma wartości bezwzględnych = 1)
        norms = np.linalg.norm(X_prepared, axis=1, ord=1)
        # Unikaj dzielenia przez zero
        norms[norms == 0] = 1.0
        X_prepared = X_prepared / norms[:, np.newaxis]
    elif normalization == 'min-max':
        # Normalizacja min-max (wartości między 0 a 1)
        for i in range(X_prepared.shape[0]):
            min_val = np.min(X_prepared[i])
            max_val = np.max(X_prepared[i])
            if max_val > min_val:
                X_prepared[i] = (X_prepared[i] - min_val) / (max_val - min_val)
            # W przypadku gdy wszystkie wartości są równe
            else:
                X_prepared[i] = np.zeros_like(X_prepared[i])
        # Dodatkowa normalizacja L2
        norms = np.linalg.norm(X_prepared, axis=1, ord=2)
        # Unikaj dzielenia przez zero
        norms[norms == 0] = 1.0
        X_prepared = X_prepared / norms[:, np.newaxis]
    
    return X_prepared

def amplitude_kernel(x1, x2):
    """
    Oblicza wartość jądra kwantowego dla dwóch wektorów przy użyciu kodowania amplitudowego.
    
    Args:
        x1, x2: Wektory cech
        
    Returns:
        Wartość jądra kwantowego
    """
    # Iloczyn skalarny kwadrat (odpowiada nakładaniu się stanów kwantowych)
    return np.abs(np.dot(x1, x2.conj()))**2

class AmplitudeKernel:
    """
    Klasa implementująca jądro kwantowe z kodowaniem amplitudowym.
    """
    def __init__(self, feature_dimension, normalization='l2'):
        """
        Inicjalizacja jądra kwantowego z kodowaniem amplitudowym.
        
        Parameters:
        -----------
        feature_dimension : int
            Liczba wymiarów wektora cech
        normalization : str
            Typ normalizacji ('l2', 'l1', lub 'min-max')
        """
        self.feature_dimension = feature_dimension
        self.normalization = normalization
        
    def evaluate(self, x1_vec, x2_vec):
        """
        Oblicza macierz jądra kwantowego dla dwóch zbiorów wektorów.
        
        Parameters:
        -----------
        x1_vec, x2_vec : ndarray
            Zbiory wektorów cech
            
        Returns:
        --------
        ndarray
            Macierz jądra kwantowego
        """
        # Przygotowanie danych
        x1_prepared = prepare_data_for_amplitude_encoding(x1_vec, self.normalization)
        x2_prepared = prepare_data_for_amplitude_encoding(x2_vec, self.normalization)
        
        # Obliczenie macierzy jądra
        kernel_matrix = np.zeros((x1_vec.shape[0], x2_vec.shape[0]))
        for i in range(x1_vec.shape[0]):
            for j in range(x2_vec.shape[0]):
                kernel_matrix[i, j] = amplitude_kernel(x1_prepared[i], x2_prepared[j])
        
        return kernel_matrix
    
    def evaluate_lists(self, x1_vec, x2_vec):
        """
        Oblicza macierz jądra kwantowego dla dwóch list wektorów.
        
        Parameters:
        -----------
        x1_vec, x2_vec : list
            Listy wektorów cech
            
        Returns:
        --------
        ndarray
            Macierz jądra kwantowego
        """
        return self.evaluate(np.array(x1_vec), np.array(x2_vec))

# Główna funkcja eksperymentalna
def run_experiments():
    # Przekierowanie wyjścia do pliku i konsoli
    sys.stdout = Logger(OUTPUT_FILE)
    
    # Informacje o środowisku
    print("======= INFORMACJE O ŚRODOWISKU =======")
    print(f"Data i czas rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Wersja NumPy: {np.__version__}")
    print(f"Wersja Pandas: {pd.__version__}")
    print(f"Wersja scikit-learn: {sklearn.__version__}")
    print(f"Wersja Qiskit: {qiskit.__version__}")
    try:
        import qiskit_machine_learning
        print(f"Wersja qiskit-machine-learning: {qiskit_machine_learning.__version__}")
    except:
        print("Nie można określić wersji qiskit-machine-learning")
    # print(f"Wersja dimod: {dimod.__version__}")  # Nie używane
    # print(f"Wersja neal: {neal.__version__}")    # Nie używane
    
    # Informacje o optymalizacjach
    print("\n======= OPTYMALIZACJE BEZPIECZEŃSTWA =======")
    print(f"Timeout operacji kwantowych: {QUANTUM_TIMEOUT} sekund")
    print(f"Liczba shots: {QUANTUM_SHOTS}")
    print(f"Maksymalny wymiar cech: {MAX_FEATURE_DIMENSION}")
    print(f"Równoległe przetwarzanie: {'TAK' if USE_PARALLEL else 'NIE'}")
    print(f"Interwał checkpoint: {CHECKPOINT_INTERVAL} iteracji")
    
    # Wyświetl informacje o wybranych eksperymentach
    print("\n======= WYBRANE EKSPERYMENTY =======")
    print(f"Eksperyment 1 (Wpływ złożoności danych): {'TAK' if RUN_COMPLEXITY_EXPERIMENT else 'NIE'}")
    print(f"Eksperyment 2 (Podzbiory genów): {'TAK' if RUN_GENE_SUBSETS_EXPERIMENT else 'NIE'}")
    print(f"Eksperyment 3 (Mapowania cech): {'TAK' if RUN_FEATURE_MAPPINGS_EXPERIMENT else 'NIE'}")
    
    # Wczytanie danych
    print("\n======= WCZYTYWANIE DANYCH =======")
    data_load_start_time = time.time()
    data = pd.read_csv(DATA_FILE)
    data_load_end_time = time.time()
    data_load_time = data_load_end_time - data_load_start_time
    print(f"Czas wczytywania danych: {data_load_time:.2f} sekund")
    print(f"Wymiary danych: {data.shape}")
    
    # Sprawdź strukturę danych
    print("\n======= STRUKTURA DANYCH =======")
    print(f"Kolumny w zbiorze danych: {data.columns.tolist()[:5]}...")
    print(f"Typy danych w kolumnach:\n{data.dtypes.head()}")
    print(f"Przykładowe dane:\n{data.head(2)}")
    
    # Sprawdź wartości w kolumnie 'Grade'
    print(f"\nUnikalne wartości w kolumnie 'Primary_Diagnosis': {data['Primary_Diagnosis'].unique()}")
    print(f"Liczba próbek dla każdej klasy:\n{data['Primary_Diagnosis'].value_counts()}")
    
    # Przygotowanie danych do eksperymentów
    print("\n======= PRZYGOTOWANIE DANYCH DO EKSPERYMENTÓW =======")
    gene_cols = [col for col in data.columns if col not in 
                ['Grade', 'Project', 'Case_ID', 'Gender', 'Age_at_diagnosis', 
                 'Primary_Diagnosis', 'Race']]
    
    # Sprawdź typ danych w kolumnach genów
    print(f"Przykładowe wartości w kolumnach genów: {data[gene_cols].iloc[0].values[:5]}")
    
    # Konwertuj dane tekstowe na binarne (0/1)
    X = data[gene_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':  # Jeśli kolumna zawiera dane tekstowe
            # Zamień 'MUTATED' na 1, a wszystko inne na 0
            X[col] = X[col].apply(lambda x: 1 if str(x).upper() == 'MUTATED' else 0)
    
    # Użyj zmiennej docelowej określonej w konfiguracji
    y = data[TARGET_VARIABLE]
    
    # Wyświetl informacje o zmiennej docelowej
    print(f"\nWybrana zmienna docelowa: {TARGET_VARIABLE}")
    print(f"Unikalne wartości w kolumnie '{TARGET_VARIABLE}': {y.unique()}")
    print(f"Liczba próbek dla każdej klasy:\n{y.value_counts()}")
    
    # Oblicz liczbę mutacji dla każdego przypadku
    mutation_counts = X.sum(axis=1)
    print(f"Średnia liczba mutacji na przypadek: {mutation_counts.mean():.2f}")
    print(f"Mediana liczby mutacji: {mutation_counts.median():.2f}")
    print(f"Min/Max liczby mutacji: {mutation_counts.min():.0f}/{mutation_counts.max():.0f}")
    
    # Inicjalizacja zmiennych do podsumowania
    complexity_results = None
    gene_subsets_results = None
    feature_mappings_results = None
    complexity_time = 0
    gene_subsets_time = 0
    feature_mappings_time = 0
    
    # Eksperyment 1: Wpływ złożoności danych
    if RUN_COMPLEXITY_EXPERIMENT:
        print("\n======= EKSPERYMENT 1: WPŁYW ZŁOŻONOŚCI DANYCH =======")
        print("UWAGA: Ten eksperyment może być bardzo czasochłonny!")
        complexity_start_time = time.time()
        complexity_results = experiment_complexity_impact(X, y, mutation_counts)
        complexity_end_time = time.time()
        complexity_time = complexity_end_time - complexity_start_time
        print(f"\nCzas wykonania eksperymentu złożoności: {complexity_time:.2f} sekund ({complexity_time/60:.2f} minut)")
        
        # Zapisz wyniki do pliku CSV
        complexity_df = pd.DataFrame(complexity_results).T
        complexity_df.to_csv(os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_zlozonosc.csv'))
        print(f"Wyniki zapisane do pliku: {os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_zlozonosc.csv')}")
    else:
        print("\n======= EKSPERYMENT 1: POMINIĘTY (zbyt czasochłonny) =======")
    
    # Eksperyment 2: Podzbiory genów
    if RUN_GENE_SUBSETS_EXPERIMENT:
        print("\n======= EKSPERYMENT 2: PODZBIORY GENÓW =======")
        print("Używając uproszczonych map cech z timeout'ami...")
        gene_subsets_start_time = time.time()
        gene_subsets_results = experiment_gene_subsets(data)
        gene_subsets_end_time = time.time()
        gene_subsets_time = gene_subsets_end_time - gene_subsets_start_time
        print(f"\nCzas wykonania eksperymentu podzbiorów genów: {gene_subsets_time:.2f} sekund ({gene_subsets_time/60:.2f} minut)")
        
        # Zapisz wyniki do pliku CSV
        gene_subsets_df = pd.DataFrame(gene_subsets_results).T
        gene_subsets_df.to_csv(os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_geny.csv'))
        print(f"Wyniki zapisane do pliku: {os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_geny.csv')}")
    else:
        print("\n======= EKSPERYMENT 2: POMINIĘTY =======")
    
    # Eksperyment 3: Mapowania cech
    if RUN_FEATURE_MAPPINGS_EXPERIMENT:
        print("\n======= EKSPERYMENT 3: MAPOWANIA CECH =======")
        print("UWAGA: Ten eksperyment może być bardzo czasochłonny!")
        # Przygotuj dane do eksperymentu mapowania cech
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        max_components = min(PCA_COMPONENTS, MAX_FEATURE_DIMENSION, X_train_scaled.shape[1])
        pca = PCA(n_components=max_components)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)
        
        print(f"Redukcja wymiarowości: {X_train_scaled.shape[1]} -> {max_components} wymiarów")
        
        feature_mappings_start_time = time.time()
        feature_mappings_results = experiment_feature_mappings(X_train_reduced, X_test_reduced, y_train, y_test)
        feature_mappings_end_time = time.time()
        feature_mappings_time = feature_mappings_end_time - feature_mappings_start_time
        print(f"\nCzas wykonania eksperymentu mapowania cech: {feature_mappings_time:.2f} sekund ({feature_mappings_time/60:.2f} minut)")
        
        # Zapisz wyniki do pliku CSV
        feature_mappings_df = pd.DataFrame({k: v['accuracy'] for k, v in feature_mappings_results.items()}, index=['accuracy']).T
        feature_mappings_df['mean_kernel'] = [v['kernel_properties']['mean'] for k, v in feature_mappings_results.items()]
        feature_mappings_df['std_kernel'] = [v['kernel_properties']['std'] for k, v in feature_mappings_results.items()]
        feature_mappings_df.to_csv(os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_mapowania.csv'))
        print(f"Wyniki zapisane do pliku: {os.path.join(OUTPUT_DIR, 'wyniki_eksperymentu_mapowania.csv')}")
    else:
        print("\n======= EKSPERYMENT 3: POMINIĘTY (problematyczny) =======")
    
    # Podsumowanie eksperymentów
    print("\n======= PODSUMOWANIE EKSPERYMENTÓW =======")
    total_time = complexity_time + gene_subsets_time + feature_mappings_time
    print(f"Całkowity czas wykonania eksperymentów: {total_time:.2f} sekund ({total_time/60:.2f} minut)")
    
    if complexity_results:
        print("\nWyniki eksperymentu złożoności danych:")
        complexity_df = pd.DataFrame(complexity_results).T
        print(complexity_df)
    
    if gene_subsets_results:
        print("\nWyniki eksperymentu podzbiorów genów:")
        gene_subsets_df = pd.DataFrame(gene_subsets_results).T
        print(gene_subsets_df)
    
    if feature_mappings_results:
        print("\nWyniki eksperymentu mapowania cech:")
        feature_mappings_df = pd.DataFrame({k: v['accuracy'] for k, v in feature_mappings_results.items()}, index=['accuracy']).T
        print(feature_mappings_df)
    
    print(f"\nWszystkie wyniki zostały zapisane w katalogu: {OUTPUT_DIR}")
    
    # Zamknięcie pliku wyjściowego
    if hasattr(sys.stdout, 'log') and not sys.stdout.log.closed:
        sys.stdout.log.close()
    sys.stdout = sys.__stdout__
    print(f"Wyniki zostały zapisane do pliku: {OUTPUT_FILE}")

# Uruchomienie eksperymentów
if __name__ == "__main__":
    run_experiments()