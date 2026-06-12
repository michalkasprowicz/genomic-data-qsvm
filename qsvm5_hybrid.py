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
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.circuit.library.data_preparation import ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import dimod

# Import funkcji z głównego modułu
import qsvm

def run_experiment():
    """
    Eksperyment 5: Podejście Hybrydowe
    Łączy wyniki wszystkich eksperymentów i używa wyżarzania symulowanego do optymalizacji
    """
    
    print("======= EKSPERYMENT 5: PODEJŚCIE HYBRYDOWE =======")
    
    # Dla każdego pliku danych
    for data_file in qsvm.DATA_FILES:
        if not os.path.exists(data_file):
            print(f"Pominięto {data_file} - plik nie istnieje")
            continue
            
        print(f"\n======= PRZETWARZANIE PLIKU: {data_file} =======")
        
        # Utwórz nazwę pliku wyjściowego
        file_base_name = os.path.basename(data_file).split('.')[0]
        output_file = os.path.join(qsvm.OUTPUT_DIR, f'wyniki_hybrid_{file_base_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
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

            # ----------------- PODEJŚCIE HYBRYDOWE -----------------
            if qsvm.RUN_HYBRID_APPROACH:
                print("\n======= PODEJŚCIE HYBRYDOWE =======")
                start_time_hybrid = time.time()
                
                # Zbierz wyniki z wszystkich eksperymentów
                all_quantum_results = []
                
                # Sprawdź cache z różnych eksperymentów
                cache_files = [
                    os.path.join(qsvm.OUTPUT_DIR, f'qsvm_zz_cache_{file_base_name}.json'),
                    os.path.join(qsvm.OUTPUT_DIR, f'qsvm_pauli_cache_{file_base_name}.json'),
                    os.path.join(qsvm.OUTPUT_DIR, f'qsvm_z_cache_{file_base_name}.json'),
                    os.path.join(qsvm.OUTPUT_DIR, f'qsvm_amplitude_cache_{file_base_name}.json')
                ]
                
                for cache_file in cache_files:
                    if os.path.exists(cache_file):
                        cache = qsvm.load_results_cache(cache_file)
                        quantum_results = cache.get('quantum_results', [])
                        all_quantum_results.extend(quantum_results)
                        print(f"Wczytano {len(quantum_results)} wyników z {cache_file}")
                
                if not all_quantum_results:
                    print("Brak wyników kwantowych do analizy hybrydowej.")
                    print("Uruchom najpierw eksperymenty 1-4.")
                    hybrid_metrics = None
                else:
                    print(f"Łącznie wczytano {len(all_quantum_results)} wyników kwantowych")
                    
                    # Wyświetl wszystkie wyniki
                    print("\nWszystkie wyniki kwantowe:")
                    for name, C, score in sorted(all_quantum_results, key=lambda x: x[2], reverse=True):
                        print(f"  {name}, C={C}: {score:.4f}")
                    
                    # Znajdź najlepszy wynik
                    best_result = max(all_quantum_results, key=lambda x: x[2])
                    print(f"\nNajlepszy wynik: {best_result[0]} z C={best_result[1]}, score={best_result[2]:.4f}")
                    
                    # ----------------- WYŻARZANIE SYMULOWANE -----------------
                    print("\n======= WYŻARZANIE SYMULOWANE =======")
                    
                    # Utworzenie problemu QUBO
                    print("Tworzenie problemu QUBO dla optymalizacji hiperparametrów...")
                    qubo_start_time = time.time()
                    
                    # Przygotuj mapy cech dla hybrydowego podejścia
                    feature_maps = []
                    feature_dimension = X_train_reduced.shape[1]
                    
                    # Dodaj wszystkie dostępne mapy cech
                    feature_maps.append({'name': 'ZZ1', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=1)})
                    feature_maps.append({'name': 'ZZ2', 'map': ZZFeatureMap(feature_dimension=feature_dimension, reps=2)})
                    feature_maps.append({'name': 'Pauli1', 'map': PauliFeatureMap(feature_dimension=feature_dimension, reps=1)})
                    feature_maps.append({'name': 'Pauli2', 'map': PauliFeatureMap(feature_dimension=feature_dimension, reps=2)})
                    feature_maps.append({'name': 'Z1', 'map': ZFeatureMap(feature_dimension=feature_dimension, reps=1)})
                    feature_maps.append({'name': 'Z2', 'map': ZFeatureMap(feature_dimension=feature_dimension, reps=2)})
                    
                    n_feature_maps = len(feature_maps)
                    n_c_values = len(qsvm.C_VALUES)
                    n_combinations = n_feature_maps * n_c_values
                    
                    # Inicjalizacja macierzy QUBO
                    Q = {}
                    
                    # Funkcja do konwersji indeksu kombinacji do cech
                    def index_to_features(idx, n_fm, n_c):
                        fm_idx = idx // n_c
                        c_idx = idx % n_c
                        return fm_idx, c_idx
                    
                    # Ewaluacja wszystkich kombinacji hiperparametrów
                    for idx in range(n_combinations):
                        fm_idx, c_idx = index_to_features(idx, n_feature_maps, n_c_values)
                        fm_name = feature_maps[fm_idx]['name']
                        C = qsvm.C_VALUES[c_idx]
                        
                        # Znajdź wynik z wcześniej obliczonych wyników
                        score = 0.0
                        for name, c_val, s in all_quantum_results:
                            if name == fm_name and c_val == C:
                                score = s
                                break
                        
                        print(f"Wynik dla {fm_name}, C={C}: {score:.4f}")
                        
                        # Ustaw wartość na diagonali (minimalizujemy -score, czyli maksymalizujemy score)
                        Q[(idx, idx)] = -score
                        
                        # Dodaj ograniczenia, aby wybrać dokładnie jedną kombinację
                        for j in range(idx + 1, n_combinations):
                            Q[(idx, j)] = qsvm.QUBO_PENALTY
                    
                    qubo_end_time = time.time()
                    qubo_time = qubo_end_time - qubo_start_time
                    print(f"Czas tworzenia problemu QUBO: {qubo_time:.2f} sekund")
                    
                    # Rozwiązywanie problemu QUBO
                    print("Rozwiązywanie problemu QUBO za pomocą wyżarzania symulowanego...")
                    annealing_start_time = time.time()
                    
                    sampler = dimod.SimulatedAnnealingSampler()
                    response = sampler.sample_qubo(Q, num_reads=qsvm.NUM_READS)
                    
                    annealing_end_time = time.time()
                    annealing_time = annealing_end_time - annealing_start_time
                    print(f"Czas wyżarzania symulowanego: {annealing_time:.2f} sekund")
                    
                    # Pobierz najlepsze rozwiązanie
                    best_solution = response.first.sample
                    best_idx = None
                    for idx, val in best_solution.items():
                        if val == 1:
                            best_idx = idx
                            break
                    
                    hybrid_accuracy = None
                    if best_idx is not None:
                        fm_idx, c_idx = index_to_features(best_idx, n_feature_maps, n_c_values)
                        best_fm_name = feature_maps[fm_idx]['name']
                        best_C = qsvm.C_VALUES[c_idx]
                        print(f"Optymalne hiperparametry z wyżarzania symulowanego: feature_map={best_fm_name}, C={best_C}")
                        
                        # Trenowanie ostatecznego modelu z optymalnymi hiperparametrami
                        try:
                            best_feature_map = feature_maps[fm_idx]['map']
                            
                            # Utworzenie quantum kernel
                            quantum_kernel_final = QuantumKernel(
                                feature_map=best_feature_map,
                                quantum_instance=ibm_backend
                            )
                            
                            # Utworzenie SVM z niestandardowym jądrem
                            def custom_kernel(X, Y):
                                return quantum_kernel_final.evaluate(X, Y)
                            
                            qsvm_final = SVC(kernel=custom_kernel, C=best_C)
                            
                            # Trenowanie i ewaluacja modelu hybrydowego
                            hybrid_train_start_time = time.time()
                            qsvm_final.fit(X_train_reduced, y_train)
                            hybrid_train_end_time = time.time()
                            hybrid_train_time = hybrid_train_end_time - hybrid_train_start_time
                            print(f"Czas trenowania modelu hybrydowego: {hybrid_train_time:.2f} sekund")
                            
                            # Ewaluacja modelu
                            hybrid_eval_start_time = time.time()
                            hybrid_pred = qsvm_final.predict(X_test_reduced)
                            hybrid_eval_end_time = time.time()
                            hybrid_eval_time = hybrid_eval_end_time - hybrid_eval_start_time
                            print(f"Czas ewaluacji modelu hybrydowego: {hybrid_eval_time:.2f} sekund")
                            
                            hybrid_accuracy = np.mean(hybrid_pred == y_test)
                            print(f"Dokładność ostatecznego modelu hybrydowego: {hybrid_accuracy:.4f}")
                            print("Raport klasyfikacji (model hybrydowy):")
                            print(classification_report(y_test, hybrid_pred, zero_division=0))
                            
                            # Zapisz szczegółowe metryki
                            hybrid_metrics = qsvm.save_metrics(y_test, hybrid_pred, "Model hybrydowy")
                        except Exception as e:
                            print(f"Błąd podczas trenowania ostatecznego modelu: {e}")
                            hybrid_metrics = None
                    else:
                        print("Nie znaleziono optymalnego rozwiązania w wyżarzaniu symulowanym.")
                        hybrid_metrics = None

                end_time_hybrid = time.time()
                hybrid_time = end_time_hybrid - start_time_hybrid
                print(f"\nCałkowity czas dla podejścia hybrydowego: {hybrid_time:.2f} sekund")
            else:
                print("\n======= PODEJŚCIE HYBRYDOWE - POMINIĘTE =======")
                hybrid_time = 0
                hybrid_metrics = None

            # ----------------- ANALIZA WYNIKÓW -----------------
            print("\n======= PORÓWNANIE WYNIKÓW =======")
            if classic_metrics:
                print(f"Klasyczny SVM: {classic_metrics['accuracy']:.4f}")
            if hybrid_metrics:
                print(f"Model hybrydowy: {hybrid_metrics['accuracy']:.4f}")

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
            print("\n======= PODSUMOWANIE EKSPERYMENTU HYBRYDOWEGO =======")
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

    print("\n======= EKSPERYMENT 5 ZAKOŃCZONY =======")

if __name__ == "__main__":
    run_experiment() 