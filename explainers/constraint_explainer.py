from pulp import *
from solvers.big_m import tighten_c1_bounds, tighten_c2_gamma

class ConstraintExplainer:
    def __init__(self, df, inputs, outputs, weight_constraints=None):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.weight_constraints = weight_constraints

    def explain_c1(self, target_id):
        """
        C1: Minimalne zbiory ograniczeń wagowych do usunięcia, aby przywrócić efektywność jednostki target_id.
        Używa ścisłych ograniczeń big-M wyliczanych przez auxiliary LP.
        """
        if not self.weight_constraints:
            return []

        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]

        # Tighten big-M bounds via auxiliary LPs
        constraint_M_bounds = tighten_c1_bounds(
            self.df, self.inputs, self.outputs, target_id, self.weight_constraints
        )

        prob = LpProblem(f"C1_Remove_Constraints_{target_id}", LpMinimize)

        # Zmienne binarne: delta_l = 1 (zostawiamy ograniczenie l), delta_l = 0 (usuwamy)
        L = len(self.weight_constraints)
        delta = LpVariable.dicts("delta", range(L), cat='Binary')

        # Wagi primalowe
        v = LpVariable.dicts("v", self.inputs, lowBound=0)
        u = LpVariable.dicts("u", self.outputs, lowBound=0)

        # Cel: Minimalizuj liczbę USUNIĘTYCH ograniczeń (czyli minimalizuj sumę 1 - delta)
        prob += lpSum([1 - delta[l] for l in range(L)])

        # Warunki efektywności targetu
        prob += lpSum([v[m] * target_dmu[m] for m in self.inputs]) == 1
        prob += lpSum([u[n] * target_dmu[n] for n in self.outputs]) == 1

        # Tradycyjne ograniczenia DEA dla wszystkich jednostek k
        for _, row in self.df.iterrows():
            prob += lpSum([u[n] * row[n] for n in self.outputs]) - \
                    lpSum([v[m] * row[m] for m in self.inputs]) <= 0

        # Ograniczenia wagowe z tightened Big-M
        M_count = len(self.inputs)
        N_count = len(self.outputs)
        for l in range(L):
            a_l = self.weight_constraints[l]
            expr_inputs = lpSum([a_l[m] * v[self.inputs[m]] for m in range(M_count)])
            expr_outputs = lpSum([a_l[M_count + n] * u[self.outputs[n]] for n in range(N_count)])
            # Jeśli delta_l = 1, to ograniczenie musi zachodzić (<= 0)
            # Jeśli delta_l = 0, to ograniczenie zostaje zrelaksowane (<= M_l)
            M_l = constraint_M_bounds.get(l, 1000.0)
            prob += expr_inputs + expr_outputs <= M_l * (1 - delta[l])

        all_removal_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            # Pobieramy usunięte ograniczenia w tym kroku (gdzie delta = 0)
            # Pobieramy usunięte ograniczenia w tym kroku (gdzie delta = 0)
            current_removed_idx = [l for l in range(L) if delta[l].varValue < 0.5]
            current_cardinality = len(current_removed_idx)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([1 - delta[l] for l in range(L)]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_removal_sets.append([f"c_{l+1}" for l in current_removed_idx])

            # Cięcie wykluczające poprzednią konfigurację usuniętych ograniczeń
            prob += lpSum([1 - delta[l] for l in current_removed_idx]) <= len(current_removed_idx) - 1

        return all_removal_sets

    def explain_c2(self, target_id, epsilon=10 ** -4):
        """
        C2: Minimalny rdzeń ograniczeń wagowych, który już sam w sobie implikuje nieefektywność.
        Używa ścisłych ograniczeń big-M wyliczanych przez auxiliary LP.
        """
        if not self.weight_constraints:
            return []

        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]

        # Tighten Γ̄_ℓ bounds via auxiliary LPs
        gamma_bounds = tighten_c2_gamma(
            self.df, self.inputs, self.outputs, target_id, self.weight_constraints, epsilon
        )

        prob = LpProblem(f"C2_Core_Constraints_{target_id}", LpMinimize)

        # Zmienne binarne: delta_l = 1 (bierzemy ograniczenie l do certyfikatu), delta_l = 0 (pomijamy)
        L = len(self.weight_constraints)
        delta = LpVariable.dicts("delta", range(L), cat='Binary')

        # Cel: Minimalizuj liczbę ograniczeń w certyfikacie
        prob += lpSum([delta[l] for l in range(L)])

        # Zmienne dualne certyfikatu
        pi = LpVariable.dicts("pi", self.df.index, lowBound=0)
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0)
        alpha = LpVariable("alpha")

        # Certyfikat nieefektywności
        prob += alpha <= 1 - epsilon

        # Powiązanie gamma_l z delta_l przy użyciu tightened Big-M
        for l in range(L):
            gamma_bound_l = gamma_bounds.get(l, 1000.0)
            prob += gamma[l] <= gamma_bound_l * delta[l]

        # Ograniczenia dualne dla wejść (m)
        M_count = len(self.inputs)
        for m_idx, m in enumerate(self.inputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][m_idx] for l in range(L)])
            prob += alpha * target_dmu[m] - lpSum([pi[k] * self.df.loc[k, m] for k in self.df.index]) + expr_gamma >= 0

        # Ograniczenia dualne dla wyjść (n)
        for n_idx, n in enumerate(self.outputs):
            expr_gamma = lpSum([gamma[l] * self.weight_constraints[l][M_count + n_idx] for l in range(L)])
            prob += lpSum([pi[k] * self.df.loc[k, n] for k in self.df.index]) + expr_gamma >= target_dmu[n]

        # Wymuszamy, by wybrany zbiór był nietrywialny (co najmniej jedno ograniczenie w rdzeniu)
        prob += lpSum([delta[l] for l in range(L)]) >= 1

        all_core_sets = []
        optimal_cardinality = None
        while True:
            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] != 'Optimal':
                break

            current_core_idx = [l for l in range(L) if delta[l].varValue > 0.5]
            current_cardinality = len(current_core_idx)

            # Po znalezieniu pierwszego rozwiązania, ograniczamy do tej samej kardynalności
            if optimal_cardinality is None:
                optimal_cardinality = current_cardinality
                prob += lpSum([delta[l] for l in range(L)]) <= optimal_cardinality
            elif current_cardinality > optimal_cardinality:
                break

            all_core_sets.append([f"c_{l+1}" for l in current_core_idx])

            # Cięcie wykluczające
            prob += lpSum([delta[l] for l in current_core_idx]) <= len(current_core_idx) - 1

        return all_core_sets