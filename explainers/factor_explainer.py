from pulp import *
from solvers.big_m import tighten_f1_bounds, tighten_f2_bounds

class FactorExplainer:
    def __init__(self, df, inputs, outputs, weight_constraints=None):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.weight_constraints = weight_constraints

    def explain_f1(self, target_id, epsilon=10 ** -4):
        """
        F1: Wszystkie minimalne podzbiory czynników wystarczające do zachowania efektywności (z ograniczeniami C).
        Używa ścisłych ograniczeń big-M wyliczanych przez auxiliary LP.
        """
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]

        # Tighten big-M bounds via auxiliary LPs
        v_bounds, u_bounds, constraint_M_bounds = tighten_f1_bounds(
            self.df, self.inputs, self.outputs, target_id, self.weight_constraints
        )

        prob = LpProblem(f"F1_Sufficient_Factors_{target_id}", LpMinimize)

        # Zmienne binarne: czy dany czynnik jest użyty?
        z_in = LpVariable.dicts("z_in", self.inputs, cat='Binary')
        z_out = LpVariable.dicts("z_out", self.outputs, cat='Binary')

        # Wagi DEA
        v = LpVariable.dicts("v", self.inputs, lowBound=0)
        u = LpVariable.dicts("u", self.outputs, lowBound=0)

        # Cel: Minimalizuj liczbę użytych czynników
        prob += lpSum([z_in[i] for i in self.inputs]) + lpSum([z_out[r] for r in self.outputs])

        # Ograniczenia modelu CCR (mnożnikowe) — Eq. F1-eff: u^T y_o = 1 (strict equality)
        prob += lpSum([v[i] * target_dmu[i] for i in self.inputs]) == 1
        prob += lpSum([u[r] * target_dmu[r] for r in self.outputs]) == 1

        for _, row in self.df.iterrows():
            prob += lpSum([u[r] * row[r] for r in self.outputs]) - \
                    lpSum([v[i] * row[i] for i in self.inputs]) <= 0

        # Powiązanie wag z zmiennymi binarnymi (tightened Big-M)
        for i in self.inputs:
            prob += v[i] <= v_bounds[i] * z_in[i]
        for r in self.outputs:
            prob += u[r] <= u_bounds[r] * z_out[r]

        # Obsługa ograniczeń wagowych C i ich deaktywacji w F1
        L = len(self.weight_constraints) if self.weight_constraints else 0
        t = LpVariable.dicts("t", range(L), cat='Binary') if L > 0 else {}
        
        M_count = len(self.inputs)
        N_count = len(self.outputs)
        
        for l in range(L):
            a_l = self.weight_constraints[l]
            supp_in = [m for m in range(M_count) if abs(a_l[m]) > 1e-9]
            supp_out = [n for n in range(N_count) if abs(a_l[M_count + n]) > 1e-9]
            
            for m in supp_in:
                prob += t[l] <= z_in[self.inputs[m]]
            for n in supp_out:
                prob += t[l] <= z_out[self.outputs[n]]
                
            expr_inputs = lpSum([a_l[m] * v[self.inputs[m]] for m in range(M_count)])
            expr_outputs = lpSum([a_l[M_count + n] * u[self.outputs[n]] for n in range(N_count)])
            M_l = constraint_M_bounds.get(l, 1000.0)
            prob += expr_inputs + expr_outputs <= M_l * (1 - t[l])

        # Co najmniej jedno wejście i jedno wyjście muszą pozostać aktywne
        prob += lpSum([z_in[i] for i in self.inputs]) >= 1
        prob += lpSum([z_out[r] for r in self.outputs]) >= 1

        all_sufficient_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            current_in = [i for i in self.inputs if z_in[i].varValue > 0.5]
            current_out = [r for r in self.outputs if z_out[r].varValue > 0.5]
            current_set = current_in + current_out
            current_cardinality = len(current_set)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([z_in[i] for i in self.inputs]) + \
                        lpSum([z_out[r] for r in self.outputs]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_sufficient_sets.append(current_set)

            # OGRANICZENIE WYKLUCZAJĄCE
            prob += lpSum([z_in[i] for i in current_in]) + \
                    lpSum([z_out[r] for r in current_out]) <= len(current_set) - 1

        return all_sufficient_sets

    def explain_f2(self, target_id, epsilon=10 ** -4):
        """
        F2: WSZYSTKIE minimalne zbiory czynników, których usunięcie powoduje utratę efektywności (z ograniczeniami C).
        Używa ścisłych ograniczeń big-M i bezpośrednio (1-z) zamiast zmiennych pomocniczych.
        """
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]

        # Tighten big-M bounds via auxiliary LPs
        g_bounds, h_bounds, gamma_bounds = tighten_f2_bounds(
            self.df, self.inputs, self.outputs, target_id, self.weight_constraints, epsilon
        )

        prob = LpProblem(f"F2_Critical_Factors_{target_id}", LpMinimize)

        # Zmienne binarne: 1 = zachowujemy czynnik, 0 = usuwamy
        z_in = LpVariable.dicts("z_in", self.inputs, cat='Binary')
        z_out = LpVariable.dicts("z_out", self.outputs, cat='Binary')

        # Zmienne dualne
        pi = LpVariable.dicts("pi", self.df.index, lowBound=0)
        alpha = LpVariable("alpha")

        # Cel: Minimalizuj liczbę USUNIĘTYCH czynników — bezpośrednio (1 - z)
        prob += lpSum([1 - z_in[i] for i in self.inputs]) + lpSum([1 - z_out[r] for r in self.outputs])

        prob += alpha <= 1 - epsilon

        # Obsługa ograniczeń wagowych C i ich deaktywacji w F2 (dual certificate)
        L = len(self.weight_constraints) if self.weight_constraints else 0
        t = LpVariable.dicts("t", range(L), cat='Binary') if L > 0 else {}
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}

        M_count = len(self.inputs)
        N_count = len(self.outputs)

        for l in range(L):
            a_l = self.weight_constraints[l]
            supp_in = [m for m in range(M_count) if abs(a_l[m]) > 1e-9]
            supp_out = [n for n in range(N_count) if abs(a_l[M_count + n]) > 1e-9]
            
            for m in supp_in:
                prob += t[l] <= z_in[self.inputs[m]]
            for n in supp_out:
                prob += t[l] <= z_out[self.outputs[n]]
                
            gamma_bound_l = gamma_bounds.get(l, 1000.0)
            prob += gamma[l] <= gamma_bound_l * t[l]

        # Ograniczenia dualne dla wejść (m) — relaxed by G_m * (1 - z_in)
        for m_idx, m in enumerate(self.inputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
            G_m = g_bounds.get(m, 1000.0)
            prob += alpha * target_dmu[m] - \
                    lpSum([pi[k] * self.df.loc[k, m] for k in self.df.index]) + expr_gamma >= -G_m * (1 - z_in[m])

        # Ograniczenia dualne dla wyjść (n) — relaxed by H_n * (1 - z_out)
        for n_idx, n in enumerate(self.outputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
            H_n = h_bounds.get(n, 1000.0)
            prob += lpSum([pi[k] * self.df.loc[k, n] for k in self.df.index]) + expr_gamma >= \
                    target_dmu[n] - H_n * (1 - z_out[n])

        # Co najmniej jedno wejście i jedno wyjście muszą pozostać aktywne
        prob += lpSum([z_in[i] for i in self.inputs]) >= 1
        prob += lpSum([z_out[r] for r in self.outputs]) >= 1

        all_critical_removal_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            removed_in = [i for i in self.inputs if z_in[i].varValue < 0.5]
            removed_out = [r for r in self.outputs if z_out[r].varValue < 0.5]
            current_removed_set = removed_in + removed_out
            current_cardinality = len(current_removed_set)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([1 - z_in[i] for i in self.inputs]) + \
                        lpSum([1 - z_out[r] for r in self.outputs]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_critical_removal_sets.append(current_removed_set)

            # OGRANICZENIE WYKLUCZAJĄCE — exclude this specific set of removed factors
            prob += lpSum([1 - z_in[i] for i in removed_in]) + \
                    lpSum([1 - z_out[r] for r in removed_out]) <= len(current_removed_set) - 1

        return all_critical_removal_sets