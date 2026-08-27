import numpy as np

def sample_feasible_weights(M, N, weight_constraints=None, n_samples=10000, max_attempts=500000):
    """
    Losuje wagi v (rozmiar M) oraz u (rozmiar N) leżące na sympleksie (sumujące się do 1),
    które spełniają ograniczenia preferencyjne weight_constraints.
    Stosuje metodę próbkowania odrzuceniowego (rejection sampling).
    """
    samples_v = []
    samples_u = []
    
    attempts = 0
    while len(samples_v) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Losowanie v na sympleksie M-wymiarowym
        if M == 1:
            v = np.array([1.0])
        else:
            v = np.random.exponential(1.0, M)
            v /= sum(v)
            
        # Losowanie u na sympleksie N-wymiarowym
        if N == 1:
            u = np.array([1.0])
        else:
            u = np.random.exponential(1.0, N)
            u /= sum(u)
            
        # Sprawdzamy ograniczenia C
        is_feasible = True
        if weight_constraints:
            w = np.concatenate([v, u])
            for a_l in weight_constraints:
                # a_l^T * w <= 0
                if np.dot(a_l, w) > 1e-9:
                    is_feasible = False
                    break
                    
        if is_feasible:
            samples_v.append(v)
            samples_u.append(u)
            
    if len(samples_v) < n_samples:
        print(f"Warning: Wygenerowano tylko {len(samples_v)} próbek na {n_samples} (max_attempts przekroczone)")
        
    return np.array(samples_v), np.array(samples_u)
