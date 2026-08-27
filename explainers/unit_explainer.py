from pulp import *
import numpy as np

class UnitExplainer:
    def __init__(self, df, inputs, outputs, weight_constraints=None):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.weight_constraints = weight_constraints

    def explain_u1(self, target_id, epsilon=10 ** -4):
        """
        U1: Minimalne zbiory świadków nieefektywności (z uwzględnieniem ograniczeń C).
        """
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]
        other_dmus = self.df[self.df['ID'] != target_id]

        prob = LpProblem(f"U1_Witness_All_{target_id}", LpMinimize)

        # Zmienne binarne: czy DMU k jest świadkiem?
        z = LpVariable.dicts("z", other_dmus.index, cat='Binary')

        # Zmienne dualne
        pi = LpVariable.dicts("pi", self.df.index, lowBound=0)
        
        # Zmienne dualne dla ograniczeń wagowych C
        L = len(self.weight_constraints) if self.weight_constraints else 0
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}
        
        alpha = LpVariable("alpha")

        # Cel: Minimalizuj liczbę wybranych świadków
        prob += lpSum([z[k] for k in other_dmus.index])

        # Ograniczenie: Efektywność musi być poniżej 1 (certyfikat nieefektywności)
        prob += alpha <= 1 - epsilon

        # Ograniczenia dualne dla wejść (m)
        for m_idx, m in enumerate(self.inputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
            prob += alpha * target_dmu[m] - lpSum([pi[k] * self.df.loc[k, m] for k in self.df.index]) + expr_gamma >= 0

        # Ograniczenia dualne dla wyjść (n)
        M_count = len(self.inputs)
        for n_idx, n in enumerate(self.outputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
            prob += lpSum([pi[k] * self.df.loc[k, n] for k in self.df.index]) + expr_gamma >= target_dmu[n]

        # Powiązanie pi_k z zmienną binarną z_k (Big-M)
        pi_bounds = self._tighten_u1_pi(target_id, epsilon)
        for k in other_dmus.index:
            prob += pi[k] <= pi_bounds[k] * z[k]

        # Wymuszamy, by wybrany zbiór świadków był niepusty
        prob += lpSum([z[k] for k in other_dmus.index]) >= 1

        # --- PĘTLA SZUKAJĄCA WSZYSTKICH ROZWIĄZAŃ O MINIMALNEJ KARDYNALNOŚCI ---
        all_witness_sets = []
        optimal_cardinality = None

        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))

            if LpStatus[status] != 'Optimal':
                break

            current_witness_indices = [k for k in other_dmus.index if z[k].varValue > 0.5]
            witness_ids = [self.df.loc[k, 'ID'] for k in current_witness_indices]
            current_cardinality = len(current_witness_indices)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([z[k] for k in other_dmus.index]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_witness_sets.append(witness_ids)

            # Ograniczenie wykluczające poprzednie rozwiązanie
            prob += lpSum([z[k] for k in current_witness_indices]) <= len(current_witness_indices) - 1

        return all_witness_sets

    def explain_u2(self, target_id):
        """
        U2: Wszystkie minimalne zbiory jednostek do usunięcia, aby przywrócić efektywność (z uwzględnieniem ograniczeń C).
        """
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]
        other_dmus = self.df[self.df['ID'] != target_id]

        prob = LpProblem(f"U2_Removal_All_{target_id}", LpMinimize)

        # Zmienne binarne: 1 = usuwamy DMU k, 0 = zostawiamy
        r = LpVariable.dicts("r", other_dmus.index, cat='Binary')

        # Wagi DEA (model mnożnikowy)
        v = LpVariable.dicts("v", self.inputs, lowBound=0)
        u = LpVariable.dicts("u", self.outputs, lowBound=0)

        # Cel: Minimalizuj liczbę usuniętych jednostek
        prob += lpSum([r[k] for k in other_dmus.index])

        # Ograniczenia: jednostka target_id musi mieć efektywność = 1
        prob += lpSum([v[m] * target_dmu[m] for m in self.inputs]) == 1
        prob += lpSum([u[n] * target_dmu[n] for n in self.outputs]) == 1

        # Ograniczenia wagowe C
        if self.weight_constraints:
            for idx, a_l in enumerate(self.weight_constraints):
                expr_inputs = lpSum([a_l[m] * v[self.inputs[m]] for m in range(len(self.inputs))])
                expr_outputs = lpSum([a_l[len(self.inputs) + n] * u[self.outputs[n]] for n in range(len(self.outputs))])
                prob += expr_inputs + expr_outputs <= 0

        # Ograniczenia dla pozostałych jednostek z Big-M (delta_bounds)
        delta_bounds = self._tighten_u2_delta(target_id)
        for k in other_dmus.index:
            row = self.df.loc[k]
            prob += lpSum([u[n] * row[n] for n in self.outputs]) <= \
                    lpSum([v[m] * row[m] for m in self.inputs]) + delta_bounds[k] * r[k]

        all_removal_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            current_removed_indices = [k for k in other_dmus.index if r[k].varValue > 0.5]
            removed_ids = [self.df.loc[k, 'ID'] for k in current_removed_indices]
            current_cardinality = len(current_removed_indices)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([r[k] for k in other_dmus.index]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_removal_sets.append(removed_ids)

            # CIĘCIE WYKLUCZAJĄCE
            prob += lpSum([r[k] for k in current_removed_indices]) <= len(current_removed_indices) - 1

        return all_removal_sets

    def explain_u3(self, target_id, smaa_data):
        """U3: Wszystkie minimalne zbiory rywali pokrywające wszystkie scenariusze."""
        beaters_per_scenario = []
        for s in smaa_data:
            target_ratio = s['ratios'][target_id]
            B_s = [k for k, val in s['ratios'].items() if k != target_id and val > target_ratio]
            if B_s:
                beaters_per_scenario.append(set(B_s))

        if not beaters_per_scenario:
            return []

        all_possible_rivals = set().union(*beaters_per_scenario)
        prob = LpProblem(f"U3_Set_Cover_{target_id}", LpMinimize)
        x = LpVariable.dicts("x", all_possible_rivals, cat='Binary')

        prob += lpSum([x[k] for k in all_possible_rivals])

        for B_s in beaters_per_scenario:
            prob += lpSum([x[k] for k in B_s]) >= 1

        # --- PĘTLA SZUKAJĄCA WSZYSTKICH ROZWIĄZAŃ O MINIMALNEJ KARDYNALNOŚCI ---
        all_cover_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            current_cover = [k for k in all_possible_rivals if x[k].varValue > 0.5]
            current_cardinality = len(current_cover)

            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([x[k] for k in all_possible_rivals]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_cover_sets.append(current_cover)

            # Cięcie wykluczające
            prob += lpSum([x[k] for k in current_cover]) <= len(current_cover) - 1

        return all_cover_sets

    def explain_u4(self, target_id, smaa_data):
        """U4: Rywale uniwersalni i obligatoryjni."""
        beaters_list = []
        for s in smaa_data:
            target_ratio = s['ratios'][target_id]
            B_s = set([k for k, val in s['ratios'].items() if k != target_id and val > target_ratio])
            if B_s:
                beaters_list.append(B_s)

        if not beaters_list:
            return {"universal": [], "mandatory": []}

        universal = set.intersection(*beaters_list)

        mandatory = set()
        for B_s in beaters_list:
            if len(B_s) == 1:
                mandatory.add(list(B_s)[0])

        return {"universal": list(universal), "mandatory": list(mandatory)}

    def explain_u5(self, smaa_data):
        """U5: Częstotliwość bycia benchmarkiem (dla jednostek efektywnych)."""
        winners = [s['winner'] for s in smaa_data]
        # Konwersja na natywne typy Pythona, aby uniknąć np.int64 w wypisywanych słownikach, jak np w robots dataset
        clean_winners = []
        for w in winners:
            if isinstance(w, (np.integer, int)):
                clean_winners.append(int(w))
            elif isinstance(w, (np.floating, float)):
                clean_winners.append(float(w))
            else:
                clean_winners.append(w)
        total = len(smaa_data)
        frequencies = {k: round(clean_winners.count(k) / total, 4) for k in set(clean_winners)}
        return frequencies

    def _tighten_u1_pi(self, target_id, epsilon=10 ** -4):
        """Wylicza max pi_k dla modelu U1 z uwzględnieniem ograniczeń C"""
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]
        pi_bounds = {}
        other_dmus = self.df[self.df['ID'] != target_id]

        L = len(self.weight_constraints) if self.weight_constraints else 0

        for k in other_dmus.index:
            prob = LpProblem(f"Tighten_Pi_{k}", LpMaximize)
            pi = LpVariable.dicts("pi", self.df.index, lowBound=0)
            gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}
            alpha = LpVariable("alpha")
            
            prob += pi[k]
            prob += alpha <= 1 - epsilon
            
            # Ograniczenia dualne dla wejść (m)
            for m_idx, m in enumerate(self.inputs):
                expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
                prob += alpha * target_dmu[m] - lpSum([pi[j] * self.df.loc[j, m] for j in self.df.index]) + expr_gamma >= 0

            # Ograniczenia dualne dla wyjść (n)
            M_count = len(self.inputs)
            for n_idx, n in enumerate(self.outputs):
                expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
                prob += lpSum([pi[j] * self.df.loc[j, n] for j in self.df.index]) + expr_gamma >= target_dmu[n]

            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
                pi_bounds[k] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
            else:
                pi_bounds[k] = 1000.0
                
        return pi_bounds

    def _tighten_u2_delta(self, target_id):
        """Wylicza Delta_k dla modelu U2 z uwzględnieniem ograniczeń C"""
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]
        delta_bounds = {}
        other_dmus = self.df[self.df['ID'] != target_id]

        for k in other_dmus.index:
            prob = LpProblem(f"Tighten_Delta_{k}", LpMaximize)
            v = LpVariable.dicts("v", self.inputs, lowBound=0)
            u = LpVariable.dicts("u", self.outputs, lowBound=0)
            
            row_k = self.df.loc[k]
            prob += lpSum([u[n] * row_k[n] for n in self.outputs]) - lpSum([v[m] * row_k[m] for m in self.inputs])
            
            prob += lpSum([v[m] * target_dmu[m] for m in self.inputs]) == 1
            prob += lpSum([u[n] * target_dmu[n] for n in self.outputs]) == 1
            
            if self.weight_constraints:
                for a_l in self.weight_constraints:
                    expr_inputs = lpSum([a_l[m] * v[self.inputs[m]] for m in range(len(self.inputs))])
                    expr_outputs = lpSum([a_l[len(self.inputs) + n] * u[self.outputs[n]] for n in range(len(self.outputs))])
                    prob += expr_inputs + expr_outputs <= 0

            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
                delta_bounds[k] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
            else:
                delta_bounds[k] = 1000.0
                
        return delta_bounds