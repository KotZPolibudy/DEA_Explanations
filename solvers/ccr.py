from pulp import *

def solve_ccr_multiplier(df, inputs, outputs, target_id, weight_constraints=None):
    """
    Rozwiązuje primalny model mnożnikowy CCR dla jednostki target_id pod wpływem ograniczeń wagowych.
    weight_constraints: Lista wektorów współczynników a_l o długości M + N, gdzie a_l^T * (v, u) <= 0.
    Zwraca: (efficiency, weights_v, weights_u)
    """
    target_dmu = df[df['ID'] == target_id].iloc[0]
    
    # Inicjalizacja problemu
    prob = LpProblem(f"CCR_Multiplier_{target_id}", LpMaximize)
    
    # Zmienne decyzyjne: wagi v dla wejść, u dla wyjść
    v = LpVariable.dicts("v", inputs, lowBound=0)
    u = LpVariable.dicts("u", outputs, lowBound=0)
    
    # Cel: Maksymalizacja ważonej sumy wyjść targetu
    prob += lpSum([u[r] * target_dmu[r] for r in outputs])
    
    # Ograniczenie normalizacyjne: ważona suma wejść targetu = 1
    prob += lpSum([v[i] * target_dmu[i] for i in inputs]) == 1
    
    # Ograniczenia dla wszystkich jednostek k: u*y_k <= v*x_k (czyli u*y_k - v*x_k <= 0)
    for _, row in df.iterrows():
        prob += lpSum([u[r] * row[r] for r in outputs]) - lpSum([v[i] * row[i] for i in inputs]) <= 0
        
    # Ograniczenia wagowe C: a_l * w <= 0
    if weight_constraints:
        for idx, a_l in enumerate(weight_constraints):
            # a_l ma długość M + N
            # Suma_{m} a_l[m]*v[m] + Suma_{n} a_l[M+n]*u[n] <= 0
            expr_inputs = lpSum([a_l[m] * v[inputs[m]] for m in range(len(inputs))])
            expr_outputs = lpSum([a_l[len(inputs) + n] * u[outputs[n]] for n in range(len(outputs))])
            prob += expr_inputs + expr_outputs <= 0
            
    # Rozwiązanie
    status = prob.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[status] == 'Optimal':
        eff = value(prob.objective)
        weights_v = {i: v[i].varValue for i in inputs}
        weights_u = {r: u[r].varValue for r in outputs}
        return eff, weights_v, weights_u
        
    return None, None, None
