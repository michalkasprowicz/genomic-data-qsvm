import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import json
import gc

# Import bibliotek kwantowych
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Import funkcji z głównego modułu
import qsvm

# Funkcja do przygotowania danych dla kodowania amplitudowego
def prepare_data_for_amplitude_encoding(data, normalization='l2'):
    """
    Przygotowuje dane dla kodowania amplitudowego z różnymi normalizacjami.
    
    Args:
        data: Dane wejściowe
        normalization: Typ normalizacji ('l2', 'l1', 'min-max')
    
    Returns:
        Przygotowane dane
    """
    if normalization == 'l2':
        # Normalizacja L2
        norms = np.linalg.norm(data, axis=1, ord=2)
        norms[norms == 0] = 1.0
        return data / norms[:, np.newaxis]
    elif normalization == 'l1':
        # Normalizacja L1
        norms = np.linalg.norm(data, axis=1, ord=1)
        norms[norms == 0] = 1.0
        return data / norms[:, np.newaxis]
    elif normalization == 'min-max':
        # Normalizacja min-max
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (data - min_vals) / range_vals
    else:
        raise ValueError(f"Nieznana normalizacja: {normalization}")

# Funkcja jądra amplitudowego
def amplitude_kernel(x1, x2):
    """
    Oblicza jądro amplitudowe między dwoma wektorami.
    
    Args:
        x1, x2: Wektory wejściowe
    
    Returns:
        Wartość jądra amplitudowego
    """
    # Oblicz iloczyn skalarny
    dot_product = np.dot(x1, x2)
    
    # Jądro amplitudowe to kwadrat iloczynu skalarnego
    return dot_product ** 2

# Klasa jądra amplitudowego
class AmplitudeKernel:
    def __init__(self, feature_dimension, normalization='l2'):
        self.feature_dimension = feature_dimension
        self.normalization = normalization
    
    def evaluate(self, x1_vec, x2_vec):
        """Oblicza macierz jądra amplitudowego"""
        # Przygotowanie danych
        x1_prepared = prepare_data_for_amplitude_encoding(x1_vec, self.normalization)
        x2_prepared = prepare_data_for_amplitude_encoding(x2_vec, self.normalization)
        
        # Obliczanie macierzy jądra
        kernel_matrix = np.zeros((x1_prepared.shape[0], x2_prepared.shape[0]))
        for i in range(x1_prepared.shape[0]):
            for j in range(x2_prepared.shape[0]):
                kernel_matrix[i, j] = amplitude_kernel(x1_prepared[i], x2_prepared[j])
        
        return kernel_matrix

def run_experiment():
    """
    Eksperyment 4: Amplitude Encoding
    Testuje klasyczny SVM i kwantowy SVM z kodowaniem amplitudowym
    """
    
    print("======= EKSPERYMENT 4: AMPLITUDE ENCODING =======")
    
    # Konfiguracja eksperymentu
    AMPLITUDE_NORMALIZATIONS = ['l2', 'l1', 'min-max']
    
    # Dla każdego pliku danych
    for data_file in qsvm.DATA_FILES:
        if not os.path.exists(data_file):
            print(f"Pominięto {data_file} - plik nie istnieje")
            continue
            
        print(f"\n======= PRZETWARZANIE PLIKU: {data_file} =======")
        
        # Utwórz nazwę pliku wyjściowego
        file_base_name = os.path.basename(data_file).split('.')[0]
        output_file = os.path.join(qsvm.OUTPUT_DIR, f'wyniki_amplitude_{file_base_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Utwórz plik cache
        cache_file = os.path.join(qsvm.OUTPUT_DIR, f'qsvm_amplitude_cache_{file_base_name}.json')
        
        # Przekierowanie wyjścia
        logger = qsvm.Logger(output_file)
        sys.stdout = logger
        
        try:
            # Przygotowanie danych
            data_dict = qsvm.prepare_data(data_file)
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            X_train_reduced = data_dict['X_train_reduced']
            X_test_reduced = data_dict['X_test_reduced']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            data_processed = data_dict['data_processed']
            
            # Inicjalizacja backendu
            ibm_service, ibm_backend, ibm_success = qsvm.initialize_ibm_quantum()
            
            # ----------------- KLASYCZNY SVM -----------------
            if qsvm.RUN_CLASSIC_SVM:
                print("\n======= KLASYCZNY SVM (BASELINE) =======")
                start_time_classic = time.time()
                
                # Trenowanie modelu
                grid = GridSearchCV(SVC(), qsvm.SVM_PARAM_GRID, cv=qsvm.SVM_CV, scoring='accuracy')
                grid.fit(X_train, y_train)
                print("Najlepsze parametry klasycznego SVM:", grid.best_params_)
                print("Dokładność klasycznego SVM:", grid.best_score_)

                # Ewaluacja modelu
                classic_pred = grid.predict(X_test)
                print("Raport klasyfikacji (klasyczny SVM):")
                print(classification_report(y_test, classic_pred, zero_division=0))

                # Zapisz szczegółowe metryki
                classic_metrics = qsvm.save_metrics(y_test, classic_pred, "Klasyczny SVM")

                end_time_classic = time.time()
                classic_svm_time = end_time_classic - start_time_classic
                print(f"\nCzas trenowania i ewaluacji klasycznego SVM: {classic_svm_time:.2f} sekund")
            else:
                print("\n======= KLASYCZNY SVM (BASELINE) - POMINIĘTY =======")
                classic_svm_time = 0
                classic_metrics = None

            # ----------------- KWANTOWY SVM Z AMPLITUDE ENCODING -----------------
            if qsvm.RUN_QUANTUM_SVM:
                print("\n======= KWANTOWY SVM Z AMPLITUDE ENCODING =======")
                start_time_quantum = time.time()
                
                # Wczytaj cache
                cache = qsvm.load_results_cache(cache_file)
                quantum_results = cache.get('quantum_results', [])

                # Testowanie każdej normalizacji
                for normalization in AMPLITUDE_NORMALIZATIONS:
                    feature_map_name = f'Amplitude_{normalization}'
                    
                    for C in qsvm.C_VALUES:
                        # Sprawdź cache
                        already_tested = False
                        for name, c_val, _ in quantum_results:
                            if name == feature_map_name and c_val == C:
                                already_tested = True
                                break
                        
                        if already_tested:
                            print(f"Pomijanie już przetestowanej kombinacji: {feature_map_name}, C={C}")
                            continue
                        
                        fm_start_time = time.time()
                        
                        try:
                            print(f"Testowanie {feature_map_name} z C={C}...")
                            
                            # Utworzenie jądra amplitudowego
                            amplitude_kernel_obj = AmplitudeKernel(
                                feature_dimension=X_train_reduced.shape[1], 
                                normalization=normalization
                            )
                            
                            # Utworzenie SVM z niestandardowym jądrem
                            def custom_kernel(X, Y):
                                return amplitude_kernel_obj.evaluate(X, Y)
                            
                            qsvm_model = SVC(kernel=custom_kernel, C=C)
                            
                            # Walidacja krzyżowa
                            cv_start_time = time.time()
                            scores = []
                            
                            kf = KFold(n_splits=qsvm.QSVM_CV, shuffle=True, random_state=qsvm.RANDOM_STATE)
                            
                            for train_idx, val_idx in kf.split(X_train_reduced):
                                X_cv_train, X_cv_val = X_train_reduced[train_idx], X_train_reduced[val_idx]
                                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                                
                                qsvm_model.fit(X_cv_train, y_cv_train)
                                score = qsvm_model.score(X_cv_val, y_cv_val)
                                scores.append(score)
                            
                            mean_score = np.mean(scores)
                            cv_end_time = time.time()
                            cv_time = cv_end_time - cv_start_time
                            
                            quantum_results.append((feature_map_name, C, mean_score))
                            
                            fm_end_time = time.time()
                            fm_time = fm_end_time - fm_start_time
                            
                            print(f"Dokładność kwantowego SVM z {feature_map_name}, C={C}: {mean_score:.4f} (czas: {fm_time:.2f} s)")
                            
                            # Zapisz wyniki pośrednie
                            cache['quantum_results'] = quantum_results
                            qsvm.save_results_cache(cache, cache_file)
                            
                        except Exception as e:
                            print(f"Błąd dla {feature_map_name}, C={C}: {str(e)}")
                            continue

                # Znajdź najlepszy model kwantowy
                if quantum_results:
                    best_qsvm = max(quantum_results, key=lambda x: x[2])
                    print(f"\nNajlepszy kwantowy SVM: {best_qsvm[0]} z C={best_qsvm[1]}, dokładność: {best_qsvm[2]:.4f}")
                    
                    # Ewaluacja najlepszego modelu
                    best_normalization = best_qsvm[0].split('_')[1]
                    print(f"Ewaluacja najlepszego modelu z kodowaniem amplitudowym (normalizacja: {best_normalization})...")
                    
                    # Utworzenie jądra kwantowego
                    amplitude_kernel_best = AmplitudeKernel(
                        feature_dimension=X_train_reduced.shape[1], 
                        normalization=best_normalization
                    )
                    
                    # Utworzenie klasyfikatora SVC z niestandardowym jądrem
                    def custom_kernel(X, Y):
                        return amplitude_kernel_best.evaluate(X, Y)
                    
                    qsvm_best = SVC(kernel=custom_kernel, C=best_qsvm[1])
                    
                    # Trenowanie modelu
                    qsvm_best.fit(X_train_reduced, y_train)
                    
                    # Ewaluacja modelu
                    quantum_pred = qsvm_best.predict(X_test_reduced)
                    print(f"Raport klasyfikacji (najlepszy kwantowy SVM z kodowaniem amplitudowym, normalizacja: {best_normalization}):")
                    print(classification_report(y_test, quantum_pred, zero_division=0))
                    
                    # Zapisz szczegółowe metryki
                    quantum_metrics = qsvm.save_metrics(y_test, quantum_pred, f"Kwantowy SVM z kodowaniem amplitudowym ({best_normalization})")
                else:
                    print("Nie udało się wytrenować żadnego modelu kwantowego.")
                    quantum_metrics = None

                end_time_quantum = time.time()
                quantum_svm_time = end_time_quantum - start_time_quantum
                print(f"\nCałkowity czas dla kwantowego SVM: {quantum_svm_time:.2f} sekund")
            else:
                print("\n======= KWANTOWY SVM - POMINIĘTY =======")
                quantum_svm_time = 0
                quantum_metrics = None

            # ----------------- ANALIZA WYNIKÓW -----------------
            print("\n======= PORÓWNANIE WYNIKÓW =======")
            if classic_metrics:
                print(f"Klasyczny SVM: {classic_metrics['accuracy']:.4f}")
            if quantum_metrics:
                print(f"Kwantowy SVM: {quantum_metrics['accuracy']:.4f}")

            # Analiza znaczenia cech (tylko dla klasycznego SVM)
            if qsvm.RUN_CLASSIC_SVM and classic_metrics:
                print("\n======= ANALIZA ZNACZENIA CECH =======")
                importance_start_time = time.time()

                result = permutation_importance(grid.best_estimator_, X_test, y_test, n_repeats=10, random_state=qsvm.RANDOM_STATE)
                important_features = []

                feature_columns = list(data_processed.columns)

                for i in range(len(feature_columns)):
                    if result.importances_mean[i] > qsvm.IMPORTANCE_THRESHOLD:
                        important_features.append((feature_columns[i], result.importances_mean[i]))

                print("Najważniejsze cechy dla klasyfikacji:")
                for feature, importance in sorted(important_features, key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {importance:.4f}")

                importance_end_time = time.time()
                importance_time = importance_end_time - importance_start_time
                print(f"\nCzas analizy znaczenia cech: {importance_time:.2f} sekund")

            # Podsumowanie
            print("\n======= PODSUMOWANIE EKSPERYMENTU AMPLITUDE =======")
            print(f"Data i czas zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            total_time = time.time() - data_dict['preparation_time']
            print(f"Całkowity czas eksperymentu: {total_time:.2f} sekund")

        except Exception as e:
            print(f"BŁĄD podczas przetwarzania {data_file}: {str(e)}")
        finally:
            # Zamknięcie pliku wyjściowego
            logger.close()
            sys.stdout = logger.terminal
            
            # Czyszczenie pamięci
            gc.collect()

    print("\n======= EKSPERYMENT 4 ZAKOŃCZONY =======")

if __name__ == "__main__":
    run_experiment() 