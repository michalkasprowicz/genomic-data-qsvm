#!/bin/bash

# Skrypt do uruchomienia eksperymentu 2 w chmurze VAST.AI
# Autor: Michał - Praca magisterska
# Data: $(date)

echo "======= URUCHAMIANIE EKSPERYMENTU 2: PODZBIORY GENÓW ======="
echo "Data rozpoczęcia: $(date)"
echo "Liczba dostępnych rdzeni CPU: $(nproc)"
echo ""

# Sprawdź czy plik experiments.py istnieje
if [ ! -f "experiments.py" ]; then
    echo "BŁĄD: Plik experiments.py nie został znaleziony!"
    echo "Upewnij się, że jesteś w odpowiednim katalogu."
    exit 1
fi

# Sprawdź czy katalog z danymi istnieje
if [ ! -d "dane" ]; then
    echo "BŁĄD: Katalog 'dane' nie został znaleziony!"
    echo "Upewnij się, że dane zostały przesłane na serwer."
    exit 1
fi

# Sprawdź czy plik z danymi istnieje
if [ ! -f "dane/TCGA_GBM_LGG_Mutations_clean.csv" ]; then
    echo "BŁĄD: Plik danych TCGA_GBM_LGG_Mutations_clean.csv nie został znaleziony!"
    echo "Sprawdź czy dane zostały poprawnie przesłane."
    exit 1
fi

echo "Wszystkie wymagane pliki zostały znalezione."
echo ""

# Sprawdź czy biblioteki są zainstalowane
echo "Sprawdzanie instalacji bibliotek..."
python3 -c "import numpy, pandas, sklearn, qiskit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalowanie wymaganych bibliotek..."
    pip install numpy pandas scikit-learn qiskit qiskit-aer qiskit-machine-learning
    if [ $? -ne 0 ]; then
        echo "BŁĄD: Nie udało się zainstalować bibliotek!"
        exit 1
    fi
else
    echo "Wszystkie biblioteki są zainstalowane."
fi

# Sprawdź czy qiskit-aer jest zainstalowany (wymagane dla nowszych wersji)
echo "Sprawdzanie qiskit-aer..."
python3 -c "import qiskit_aer" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalowanie qiskit-aer..."
    pip install qiskit-aer
    if [ $? -ne 0 ]; then
        echo "OSTRZEŻENIE: Nie udało się zainstalować qiskit-aer, ale eksperyment może działać z starszą wersją Qiskit"
    fi
fi

echo ""
echo "Uruchamianie eksperymentu..."
echo ""

# Uruchom eksperyment
python3 experiments.py

# Sprawdź czy eksperyment zakończył się sukcesem
if [ $? -eq 0 ]; then
    echo ""
    echo "======= EKSPERYMENT ZAKOŃCZONY SUKCESEM ======="
    echo "Data zakończenia: $(date)"
    echo ""
    echo "Wyniki zostały zapisane w katalogu: wyniki/eksperyment_2_podzbiory_genow/"
    echo ""
    echo "Sprawdź pliki wyników:"
    ls -la wyniki/eksperyment_2_podzbiory_genow/ 2>/dev/null || echo "Katalog wyników nie został utworzony."
else
    echo ""
    echo "======= EKSPERYMENT ZAKOŃCZONY BŁĘDEM ======="
    echo "Data zakończenia: $(date)"
    echo "Sprawdź logi powyżej aby zidentyfikować problem."
    exit 1
fi
