import pandas as pd
import numpy as np
# Dane przepisywane przez LLM, bo tak było szybciej!


# Lotniska sprawdzone ręcznie.
def get_airport_data():
    data = {
        'ID': ['WAW', 'KRK', 'KAT', 'WRO', 'POZ', 'LCJ', 'GDN', 'SZZ', 'BZG', 'RZE', 'IEG'],
        'i1': [10.5, 3.1, 3.6, 1.5, 1.5, 0.6, 1.0, 0.7, 0.3, 0.6, 0.1],
        'i2': [36, 19, 32, 12, 10, 12, 15, 10, 6, 6, 10],
        'i3': [129.4, 31.6, 57.6, 18.0, 24.0, 24.0, 42.9, 25.7, 3.4, 11.3, 63.4],
        'i4': [7.0, 7.9, 10.5, 3.0, 4.0, 3.9, 2.5, 1.9, 1.2, 2.7, 3.0],
        'o1': [9.5, 2.9, 2.4, 1.5, 1.3, 0.3, 2.0, 0.3, 0.3, 0.3, 0.005],
        'o2': [129.7, 31.3, 21.1, 18.8, 16.2, 4.2, 23.6, 4.2, 4.2, 3.5, 0.61],
        'is_efficient': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    }
    return pd.DataFrame(data), ['i1', 'i2', 'i3', 'i4'], ['o1', 'o2']


# Sprawdzone ręcznie
def get_robot_data():
    data = {
        'ID': range(1, 28),
        'Cost': [7.2, 4.8, 5.0, 7.2, 9.6, 1.07, 1.76, 3.2, 6.72, 2.4, 2.88, 6.9, 3.2, 4.0, 3.68, 6.88, 8.0, 6.3, 0.94, 0.16, 2.81, 3.8, 1.25, 1.37, 3.63, 5.3, 4.0],
        'Load': [60, 6, 45, 1.5, 50, 1, 5, 15, 10, 6, 30, 13.6, 10, 30, 47, 80, 15, 10, 10, 1.5, 27, 0.9, 2.5, 2.5, 10, 70, 205],
        'Velocity': [1.35, 1.1, 1.27, 0.66, 0.05, 0.3, 1.0, 1.0, 1.1, 1.0, 0.9, 0.15, 1.2, 1.2, 1.0, 1.0, 2.0, 1.0, 0.3, 0.8, 1.7, 1.0, 0.5, 0.5, 1.0, 1.25, 0.75],
        'Repeat': [0.15, 0.05, 1.27, 0.025, 0.25, 0.1, 0.1, 0.1, 0.2, 0.05, 0.5, 1.0, 0.05, 0.05, 1.0, 1.0, 2.0, 0.2, 0.05, 2.0, 2.0, 0.05, 0.1, 0.1, 0.2, 1.27, 2.03],
        'eff_score': [1.0, 0.9, 0.53, 1.0, 0.59, 0.48, 1.0, 0.78, 0.38, 1.0, 0.67, 0.1, 1.0, 1.0, 0.61, 0.61, 0.41, 0.37, 1.0, 1.0, 0.85, 0.83, 0.69, 0.64, 0.55, 0.58, 1.0]
    }
    df = pd.DataFrame(data)
    df['is_efficient'] = (df['eff_score'] == 1.0).astype(int)  # check czy to jest tak samo jak to co java powie
    return df, ['Cost', 'Repeat'], ['Load', 'Velocity']


# Sprawdzone ręcznie, ale java ma zupełnie inny przykład szpitali - potrzebny własny model
def get_hospital_data():
    data = {
        'ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
        'Doctors': [20, 19, 25, 27, 22, 55, 33, 31, 30, 50, 53, 38],
        'Nurses': [151, 131, 160, 168, 158, 255, 235, 206, 244, 268, 306, 284],
        'Outpatients': [100, 150, 160, 180, 94, 230, 220, 152, 190, 250, 260, 250],
        'Inpatients': [90, 50, 55, 72, 66, 90, 88, 80, 100, 100, 147, 120]
    }
    df = pd.DataFrame(data)
    df['is_efficient'] = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
    return df, ['Doctors', 'Nurses'], ['Outpatients', 'Inpatients']


def load_smaa_results(df, inputs, outputs, input_weights_path, output_weights_path):
    """
    Ładowanie autentycznych wag SMAA z plików CSV.
    Dla każdego scenariusza obliczany jest stosunek u*y / v*x dla każdej jednostki k
    oraz wyznaczany jest zwycięzca (lider).
    """
    import os
    if not os.path.exists(input_weights_path) or not os.path.exists(output_weights_path):
        raise FileNotFoundError(f"Nie znaleziono pliku wag: {input_weights_path} lub {output_weights_path}")
        
    input_weights = pd.read_csv(input_weights_path)
    output_weights = pd.read_csv(output_weights_path)
    
    n_scenarios = min(len(input_weights), len(output_weights))
    orig_id_type = type(df['ID'].iloc[0])
    
    scenarios = []
    for s in range(n_scenarios):
        v = input_weights.iloc[s].values
        u = output_weights.iloc[s].values
        
        if len(v) != len(inputs):
            raise ValueError(f"Liczba wag wejściowych ({len(v)}) nie odpowiada liczbie wejść ({len(inputs)}) w scenariuszu {s}")
        if len(u) != len(outputs):
            raise ValueError(f"Liczba wag wyjściowych ({len(u)}) nie odpowiada liczbie wyjść ({len(outputs)}) w scenariuszu {s}")
            
        ratios = {}
        for _, row in df.iterrows():
            weighted_output = sum(u[i] * row[outputs[i]] for i in range(len(outputs)))
            weighted_input = sum(v[i] * row[inputs[i]] for i in range(len(inputs)))
            ratios[orig_id_type(row['ID'])] = weighted_output / weighted_input if weighted_input > 1e-12 else 0.0
            
        scenarios.append({
            'weights_v': v,
            'weights_u': u,
            'ratios': ratios,
            'winner': orig_id_type(max(ratios, key=ratios.get))
        })
        
    return scenarios


def get_running_example_data():
    """
    Zwraca zbiór danych Running Example z dea-explanations.tex (Tabela 1).
    10 DMUs (A-J), 2 wejścia (x1, x2), 2 wyjścia (y1, y2).
    """
    data = {
        'ID': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'x1': [8.0, 10.0, 9.0, 9.0, 10.0, 7.0, 12.0, 11.0, 8.0, 6.0],
        'x2': [9.0, 9.0, 10.0, 8.0, 10.0, 11.0, 7.0, 11.0, 8.0, 12.0],
        'y1': [5.0, 6.0, 5.0, 4.0, 5.0, 4.0, 7.0, 6.0, 4.0, 1.0],
        'y2': [2.0, 2.0, 3.0, 2.0, 2.0, 4.0, 1.0, 3.0, 1.0, 7.0]
    }
    df = pd.DataFrame(data)
    # Etykiety is_efficient ustawiane przez base.py (efficient_ids), nie tutaj.
    return df, ['x1', 'x2'], ['y1', 'y2']


def sample_smaa_results(df, inputs, outputs, weight_constraints=None, n_scenarios=1000):
    """
    Generuje próbki SMAA na podstawie próbkowania z ograniczeniami z sampling.py.
    """
    import sampling
    M = len(inputs)
    N = len(outputs)
    
    samples_v, samples_u = sampling.sample_feasible_weights(M, N, weight_constraints, n_scenarios)
    orig_id_type = type(df['ID'].iloc[0])
    
    scenarios = []
    for s in range(len(samples_v)):
        v = samples_v[s]
        u = samples_u[s]
        
        ratios = {}
        for _, row in df.iterrows():
            weighted_output = sum(u[i] * row[outputs[i]] for i in range(N))
            weighted_input = sum(v[i] * row[inputs[i]] for i in range(M))
            ratios[orig_id_type(row['ID'])] = weighted_output / weighted_input if weighted_input > 1e-12 else 0.0
            
        scenarios.append({
            'weights_v': v,
            'weights_u': u,
            'ratios': ratios,
            'winner': orig_id_type(max(ratios, key=ratios.get))
        })
    return scenarios



