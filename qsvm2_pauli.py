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
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Import funkcji z głównego modułu
import qsvm

def run_experiment():
    """
    Eksperyment 2: Pauli1 i Pauli2 Feature Maps
    Testuje klasyczny SVM i kwantowy SVM z mapami cech Pauli1 i Pauli2
    """
    
    print("======= EKSPERYMENT 2: PAULI1 i PAULI2 FEATURE MAPS =======")
    
    # Konfiguracja eksperymentu
    FEATURE_MAPS = {
        'Pauli1': {'reps': 1, 'enabled': True},
        'Pauli2': {'reps': 2, 'enabled': True}
    }
    
    # Dla każdego pliku danych
    for data_file in qsvm.DATA_FILES:
        if not os.path.exists(data_file):
            print(f"Pominięto {data_file} - plik nie istnieje")
            continue
            
        print(f"\n======= PRZETWARZANIE PLIKU: {data_file} =======")
        
        # Utwórz nazwę pliku wyjściowego
        file_base_name = os.path.basename(data_file).split('.')[0]
        output_file = os.path.join(qsvm.OUTPUT_DIR, f'wyniki_pauli_{file_base_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Utwórz plik cache
        cache_file = os.path.join(qsvm.OUTPUT_DIR, f'qsvm_pauli_cache_{file_base_name}.json')
        
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

            # ----------------- KWANTOWY SVM -----------------
            if qsvm.RUN_QUANTUM_SVM:
                print("\n======= KWANTOWY SVM Z PAULI FEATURE MAPS =======")
                start_time_quantum = time.time()
                
                # Wczytaj cache
                cache = qsvm.load_results_cache(cache_file)
                quantum_results = cache.get('quantum_results', [])

                # Tworzenie map cech
                feature_maps = []
                feature_dimension = X_train_reduced.shape[1]
                
                for name, config in FEATURE_MAPS.items():
                    if config['enabled']:
                        feature_map = PauliFeatureMap(feature_dimension=feature_dimension, reps=config['reps'])
                        feature_maps.append({'name': name, 'map': feature_map})

                print(f"Testowanie {len(feature_maps)} map cech: {[fm['name'] for fm in feature_maps]}")

                # Testowanie każdej mapy cech
                for fm in feature_maps:
                    for C in qsvm.C_VALUES:
                        # Sprawdź cache
                        already_tested = False
                        for name, c_val, _ in quantum_results:
                            if name == fm['name'] and c_val == C:
                                already_tested = True
                                break
                        
                        if already_tested:
                            print(f"Pomijanie już przetestowanej kombinacji: {fm['name']}, C={C}")
                            continue
                        
                        fm_start_time = time.time()
                        
                        try:
                            print(f"Testowanie {fm['name']} z C={C}...")
                            
                            # Debugowanie danych
                            print(f"  Wymiary danych: X_train_reduced {X_train_reduced.shape}")
                            print(f"  Sprawdzenie NaN: {np.isnan(X_train_reduced).sum()}")
                            print(f"  Sprawdzenie inf: {np.isinf(X_train_reduced).sum()}")
                            print(f"  Zakres danych: [{X_train_reduced.min():.4f}, {X_train_reduced.max():.4f}]")
                            
                            # Utworzenie quantum kernel z debugowaniem
                            quantum_kernel = QuantumKernel(
                                feature_map=fm['map'],
                                quantum_instance=ibm_backend
                            )
                            
                            # Test quantum kernel
                            print(f"  Testowanie quantum kernel...")
                            try:
                                test_kernel = quantum_kernel.evaluate(X_train_reduced[:2], X_train_reduced[:2])
                                print(f"  Test kernel shape: {test_kernel.shape}")
                                print(f"  Test kernel range: [{test_kernel.min():.4f}, {test_kernel.max():.4f}]")
                                if np.isnan(test_kernel).any() or np.isinf(test_kernel).any():
                                    print(f"  BŁĄD: Kernel zawiera NaN lub inf!")
                                    continue
                            except Exception as e:
                                print(f"  BŁĄD testowania kernel: {str(e)}")
                                continue
                            
                            # Utworzenie SVM z niestandardowym jądrem i debugowaniem
                            def custom_kernel(X, Y):
                                try:
                                    kernel_matrix = quantum_kernel.evaluate(X, Y)
                                    # Sprawdź czy kernel jest poprawny
                                    if np.isnan(kernel_matrix).any() or np.isinf(kernel_matrix).any():
                                        print(f"    BŁĄD: Kernel matrix zawiera NaN lub inf!")
                                        return np.eye(len(X), len(Y))  # Fallback
                                    return kernel_matrix
                                except Exception as e:
                                    print(f"    BŁĄD kernel evaluation: {str(e)}")
                                    return np.eye(len(X), len(Y))  # Fallback
                            
                            qsvm_model = SVC(kernel=custom_kernel, C=C, random_state=qsvm.RANDOM_STATE)
                            
                            # Walidacja krzyżowa z debugowaniem
                            cv_start_time = time.time()
                            scores = []
                            
                            kf = KFold(n_splits=qsvm.QSVM_CV, shuffle=True, random_state=qsvm.RANDOM_STATE)
                            
                            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_reduced)):
                                X_cv_train, X_cv_val = X_train_reduced[train_idx], X_train_reduced[val_idx]
                                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                                
                                print(f"    Fold {fold+1}/{qsvm.QSVM_CV}: train {X_cv_train.shape}, val {X_cv_val.shape}")
                                
                                try:
                                    qsvm_model.fit(X_cv_train, y_cv_train)
                                    score = qsvm_model.score(X_cv_val, y_cv_val)
                                    scores.append(score)
                                    print(f"    Fold {fold+1} score: {score:.4f}")
                                except Exception as e:
                                    print(f"    BŁĄD fold {fold+1}: {str(e)}")
                                    scores.append(0.0)  # Fallback
                            
                            if len(scores) > 0:
                                mean_score = np.mean(scores)
                                std_score = np.std(scores)
                                print(f"    Wszystkie scores: {scores}")
                                print(f"    Mean score: {mean_score:.4f} ± {std_score:.4f}")
                            else:
                                mean_score = 0.0
                                print(f"    BŁĄD: Brak poprawnych scores!")
                            
                            cv_end_time = time.time()
                            cv_time = cv_end_time - cv_start_time
                            
                            quantum_results.append((fm['name'], C, mean_score))
                            
                            fm_end_time = time.time()
                            fm_time = fm_end_time - fm_start_time
                            
                            print(f"Dokładność kwantowego SVM z {fm['name']}, C={C}: {mean_score:.4f} (czas: {fm_time:.2f} s)")
                            
                            # Zapisz wyniki pośrednie
                            cache['quantum_results'] = quantum_results
                            qsvm.save_results_cache(cache, cache_file)
                            
                        except Exception as e:
                            print(f"BŁĄD dla {fm['name']}, C={C}: {str(e)}")
                            # Dodaj fallback wynik
                            quantum_results.append((fm['name'], C, 0.0))
                            cache['quantum_results'] = quantum_results
                            qsvm.save_results_cache(cache, cache_file)
                            continue

                # Znajdź najlepszy model kwantowy
                if quantum_results:
                    best_qsvm = max(quantum_results, key=lambda x: x[2])
                    print(f"\nNajlepszy kwantowy SVM: {best_qsvm[0]} z C={best_qsvm[1]}, dokładność: {best_qsvm[2]:.4f}")
                    
                    # Ewaluacja najlepszego modelu
                    best_feature_map = None
                    for fm in feature_maps:
                        if fm['name'] == best_qsvm[0]:
                            best_feature_map = fm['map']
                            break
                    
                    if best_feature_map:
                        quantum_kernel_best = QuantumKernel(
                            feature_map=best_feature_map,
                            quantum_instance=ibm_backend
                        )
                        
                        qsvm_best = SVC(kernel=quantum_kernel_best.evaluate, C=best_qsvm[1])
                        qsvm_best.fit(X_train_reduced, y_train)
                        
                        quantum_pred = qsvm_best.predict(X_test_reduced)
                        print("Raport klasyfikacji (najlepszy kwantowy SVM):")
                        print(classification_report(y_test, quantum_pred, zero_division=0))
                        
                        quantum_metrics = qsvm.save_metrics(y_test, quantum_pred, f"Kwantowy SVM {best_qsvm[0]}")
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
            print("\n======= PODSUMOWANIE EKSPERYMENTU PAULI =======")
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

    print("\n======= EKSPERYMENT 2 ZAKOŃCZONY =======")

if __name__ == "__main__":
    run_experiment() 