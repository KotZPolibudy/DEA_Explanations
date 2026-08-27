from pulp import *


def tighten_f1_bounds(df, inputs, outputs, target_id, weight_constraints=None):
    """
    Tightens big-M bounds for F1 (Minimal Sufficient Factor Subset).
    Computes V̄_m (max v_m), Ū_n (max u_n), and M_ℓ (max a_ℓ^T w)
    under the base feasibility conditions: v^T x_o = 1, u^T y_k ≤ v^T x_k ∀k, v,u ≥ 0.
    Note: weight constraints are NOT enforced here because F1 may deactivate them.
    """
    target_dmu = df[df['ID'] == target_id].iloc[0]

    # --- V̄_m bounds ---
    v_bounds = {}
    for m in inputs:
        prob = LpProblem(f"F1_Vbar_{m}", LpMaximize)
        v = LpVariable.dicts("v", inputs, lowBound=0)
        u = LpVariable.dicts("u", outputs, lowBound=0)

        prob += v[m]
        prob += lpSum([v[i] * target_dmu[i] for i in inputs]) == 1

        for _, row in df.iterrows():
            prob += lpSum([u[r] * row[r] for r in outputs]) <= lpSum([v[i] * row[i] for i in inputs])

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            v_bounds[m] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            v_bounds[m] = 1000.0

    # --- Ū_n bounds ---
    u_bounds = {}
    for n in outputs:
        prob = LpProblem(f"F1_Ubar_{n}", LpMaximize)
        v = LpVariable.dicts("v", inputs, lowBound=0)
        u = LpVariable.dicts("u", outputs, lowBound=0)

        prob += u[n]
        prob += lpSum([v[i] * target_dmu[i] for i in inputs]) == 1

        for _, row in df.iterrows():
            prob += lpSum([u[r] * row[r] for r in outputs]) <= lpSum([v[i] * row[i] for i in inputs])

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            u_bounds[n] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            u_bounds[n] = 1000.0

    # --- M_ℓ bounds (max a_ℓ^T w under base feasibility) ---
    M_count = len(inputs)
    constraint_bounds = {}
    if weight_constraints:
        for l, a_l in enumerate(weight_constraints):
            prob = LpProblem(f"F1_Mbar_{l}", LpMaximize)
            v = LpVariable.dicts("v", inputs, lowBound=0)
            u = LpVariable.dicts("u", outputs, lowBound=0)

            expr_inputs = lpSum([a_l[m] * v[inputs[m]] for m in range(M_count)])
            expr_outputs = lpSum([a_l[M_count + n] * u[outputs[n]] for n in range(len(outputs))])
            prob += expr_inputs + expr_outputs

            prob += lpSum([v[i] * target_dmu[i] for i in inputs]) == 1
            for _, row in df.iterrows():
                prob += lpSum([u[r] * row[r] for r in outputs]) <= lpSum([v[i] * row[i] for i in inputs])

            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
                constraint_bounds[l] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
            else:
                constraint_bounds[l] = 1000.0

    return v_bounds, u_bounds, constraint_bounds


def tighten_f2_bounds(df, inputs, outputs, target_id, weight_constraints=None, epsilon=1e-4):
    """
    Tightens big-M bounds for F2 (Minimal Factor Removal Destroying Efficiency).
    Computes G_m, H_n (relaxation constants for dual constraints when factors are removed)
    and Γ̄_ℓ (max gamma_ℓ in dual certificate).
    All computed under the dual feasibility conditions with α ≤ 1 − ε.
    """
    target_dmu = df[df['ID'] == target_id].iloc[0]
    L = len(weight_constraints) if weight_constraints else 0
    M_count = len(inputs)
    N_count = len(outputs)

    def _build_base_dual(prob_name, obj_var_name=None):
        """Build base dual feasibility LP for F2 tightening."""
        prob = LpProblem(prob_name, LpMaximize)
        pi = LpVariable.dicts("pi", df.index, lowBound=0)
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}
        alpha = LpVariable("alpha")

        prob += alpha <= 1 - epsilon

        # Dual input constraints (all enforced — no relaxation during tightening)
        for m_idx, m in enumerate(inputs):
            expr_gamma = lpSum([gamma[l] * weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
            prob += alpha * target_dmu[m] - lpSum([pi[k] * df.loc[k, m] for k in df.index]) + expr_gamma >= 0

        # Dual output constraints (all enforced)
        for n_idx, n in enumerate(outputs):
            expr_gamma = lpSum([gamma[l] * weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
            prob += lpSum([pi[k] * df.loc[k, n] for k in df.index]) + expr_gamma >= target_dmu[n]

        return prob, pi, gamma, alpha

    # --- G_m bounds (max LHS violation when input m is removed) ---
    g_bounds = {}
    for m_idx, m in enumerate(inputs):
        prob = LpProblem(f"F2_G_{m}", LpMaximize)
        pi = LpVariable.dicts("pi", df.index, lowBound=0)
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}
        alpha = LpVariable("alpha")

        prob += alpha <= 1 - epsilon

        # Max the violation of the m-th dual constraint
        expr_gamma_m = lpSum([gamma[l] * weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
        lhs_m = alpha * target_dmu[m] - lpSum([pi[k] * df.loc[k, m] for k in df.index]) + expr_gamma_m
        prob += -lhs_m  # maximize negative of LHS = maximize violation

        # Enforce all other dual constraints
        for m2_idx, m2 in enumerate(inputs):
            if m2 != m:
                expr_gamma2 = lpSum([gamma[l] * weight_constraints[l][m2_idx] for l in range(L)]) if L > 0 else 0
                prob += alpha * target_dmu[m2] - lpSum([pi[k] * df.loc[k, m2] for k in df.index]) + expr_gamma2 >= 0

        for n_idx, n in enumerate(outputs):
            expr_gamma_n = lpSum([gamma[l] * weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
            prob += lpSum([pi[k] * df.loc[k, n] for k in df.index]) + expr_gamma_n >= target_dmu[n]

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            g_bounds[m] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            g_bounds[m] = 1000.0

    # --- H_n bounds (max LHS violation when output n is removed) ---
    h_bounds = {}
    for n_idx, n in enumerate(outputs):
        prob = LpProblem(f"F2_H_{n}", LpMaximize)
        pi = LpVariable.dicts("pi", df.index, lowBound=0)
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0) if L > 0 else {}
        alpha = LpVariable("alpha")

        prob += alpha <= 1 - epsilon

        # Max the shortfall of the n-th dual constraint
        expr_gamma_n = lpSum([gamma[l] * weight_constraints[l][M_count + n_idx] for l in range(L)]) if L > 0 else 0
        rhs_n = target_dmu[n]
        lhs_n = lpSum([pi[k] * df.loc[k, n] for k in df.index]) + expr_gamma_n
        prob += rhs_n - lhs_n  # maximize how much RHS exceeds LHS

        # Enforce all input dual constraints
        for m_idx, m in enumerate(inputs):
            expr_gamma_m = lpSum([gamma[l] * weight_constraints[l][m_idx] for l in range(L)]) if L > 0 else 0
            prob += alpha * target_dmu[m] - lpSum([pi[k] * df.loc[k, m] for k in df.index]) + expr_gamma_m >= 0

        # Enforce all other output dual constraints
        for n2_idx, n2 in enumerate(outputs):
            if n2 != n:
                expr_gamma_n2 = lpSum([gamma[l] * weight_constraints[l][M_count + n2_idx] for l in range(L)]) if L > 0 else 0
                prob += lpSum([pi[k] * df.loc[k, n2] for k in df.index]) + expr_gamma_n2 >= target_dmu[n2]

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            h_bounds[n] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            h_bounds[n] = 1000.0

    # --- Γ̄_ℓ bounds (max gamma_ℓ in dual certificate) ---
    gamma_bounds = {}
    if weight_constraints:
        for l_target in range(L):
            prob, pi, gamma, alpha = _build_base_dual(f"F2_Gamma_{l_target}")
            prob += gamma[l_target]

            status = prob.solve(PULP_CBC_CMD(msg=0))
            if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
                gamma_bounds[l_target] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
            else:
                gamma_bounds[l_target] = 1000.0

    return g_bounds, h_bounds, gamma_bounds


def tighten_c1_bounds(df, inputs, outputs, target_id, weight_constraints):
    """
    Tightens M_ℓ for C1 (Constraint Removal to Restore Efficiency).
    M_ℓ = max{a_ℓ^T(v,u) : v^T x_o = 1, u^T y_o = 1, u^T y_k ≤ v^T x_k ∀k, v,u ≥ 0}.
    """
    if not weight_constraints:
        return {}

    target_dmu = df[df['ID'] == target_id].iloc[0]
    M_count = len(inputs)
    constraint_bounds = {}

    for l, a_l in enumerate(weight_constraints):
        prob = LpProblem(f"C1_Mbar_{l}", LpMaximize)
        v = LpVariable.dicts("v", inputs, lowBound=0)
        u = LpVariable.dicts("u", outputs, lowBound=0)

        expr_inputs = lpSum([a_l[m] * v[inputs[m]] for m in range(M_count)])
        expr_outputs = lpSum([a_l[M_count + n] * u[outputs[n]] for n in range(len(outputs))])
        prob += expr_inputs + expr_outputs

        prob += lpSum([v[m] * target_dmu[m] for m in inputs]) == 1
        prob += lpSum([u[n] * target_dmu[n] for n in outputs]) == 1

        for _, row in df.iterrows():
            prob += lpSum([u[n] * row[n] for n in outputs]) <= lpSum([v[m] * row[m] for m in inputs])

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            constraint_bounds[l] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            constraint_bounds[l] = 1000.0

    return constraint_bounds


def tighten_c2_gamma(df, inputs, outputs, target_id, weight_constraints, epsilon=1e-4):
    """
    Tightens Γ̄_ℓ for C2 (Core Constraint Subset).
    For each ℓ, max gamma_ℓ subject to dual feasibility with α ≤ 1 − ε.
    """
    if not weight_constraints:
        return {}

    target_dmu = df[df['ID'] == target_id].iloc[0]
    L = len(weight_constraints)
    M_count = len(inputs)
    gamma_bounds = {}

    for l_target in range(L):
        prob = LpProblem(f"C2_Gamma_{l_target}", LpMaximize)
        pi = LpVariable.dicts("pi", df.index, lowBound=0)
        gamma = LpVariable.dicts("gamma", range(L), lowBound=0)
        alpha = LpVariable("alpha")

        prob += gamma[l_target]
        prob += alpha <= 1 - epsilon

        # Dual input constraints
        for m_idx, m in enumerate(inputs):
            expr_gamma = lpSum([gamma[l] * weight_constraints[l][m_idx] for l in range(L)])
            prob += alpha * target_dmu[m] - lpSum([pi[k] * df.loc[k, m] for k in df.index]) + expr_gamma >= 0

        # Dual output constraints
        for n_idx, n in enumerate(outputs):
            expr_gamma = lpSum([gamma[l] * weight_constraints[l][M_count + n_idx] for l in range(L)])
            prob += lpSum([pi[k] * df.loc[k, n] for k in df.index]) + expr_gamma >= target_dmu[n]

        status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[status] == 'Optimal' and value(prob.objective) is not None:
            gamma_bounds[l_target] = max(1e-4, value(prob.objective) * 1.1 + 0.1)
        else:
            gamma_bounds[l_target] = 1000.0

    return gamma_bounds
