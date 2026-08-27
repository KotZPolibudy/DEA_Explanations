import utils.data_loader as data_loader
from core.explainer import DEAXplainer
import pandas as pd
import sys
import os
import numpy as np
from solvers.ccr import solve_ccr_multiplier

sys.stdout.reconfigure(encoding='utf-8')


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def run_analysis_for_dataset(get_data_func, dataset_name, input_weights_path=None, output_weights_path=None, efficient_ids=None, weight_constraints=None):
    filename = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = filename.replace("ó", "o").replace("ł", "l").replace("ś", "s").replace("ć", "c").replace("ź", "z").replace("ń", "n").replace("ą", "a").replace("ę", "e")
    filename += ".txt"
    output_dir = "DEA_explanations_output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f_out:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f_out)
        try:
            _run_analysis_for_dataset_impl(get_data_func, dataset_name, input_weights_path, output_weights_path, efficient_ids, weight_constraints)
        finally:
            sys.stdout = original_stdout


def _run_analysis_for_dataset_impl(get_data_func, dataset_name, input_weights_path=None, output_weights_path=None, efficient_ids=None, weight_constraints=None):
    df, inputs, outputs = get_data_func()
    
    if efficient_ids is not None:
        df['is_efficient'] = df['ID'].apply(lambda x: 1 if x in efficient_ids else 0)
        
    xplainer = DEAXplainer(df, inputs, outputs, dataset_name, weight_constraints)

    # Ładowanie autentycznych wyników SMAA dla datasetu lub losowanie dynamiczne z ograniczeniami
    if input_weights_path and output_weights_path:
        smaa_results = data_loader.load_smaa_results(df, inputs, outputs, input_weights_path, output_weights_path)
    else:
        smaa_results = data_loader.sample_smaa_results(df, inputs, outputs, weight_constraints, n_scenarios=10000)

    # Augmentacja próbek wagowych SMAA, aby każda matematycznie efektywna jednostka była liderem przynajmniej raz
    winners_in_smaa = set(s['winner'] for s in smaa_results)
    efficient_ids_in_df = df[df['is_efficient'] == 1]['ID'].tolist()
    orig_id_type = type(df['ID'].iloc[0])
    
    for target in efficient_ids_in_df:
        if target not in winners_in_smaa:
            # Obliczamy optymalne wagi przez LP CCR pod wpływem ograniczeń
            eff, v_dict, u_dict = solve_ccr_multiplier(df, inputs, outputs, target, weight_constraints)
            if eff is not None and eff >= 1.0 - 1e-5:
                v = np.array([v_dict[i] for i in inputs])
                u = np.array([u_dict[r] for r in outputs])
                
                v_sum = sum(v)
                u_sum = sum(u)
                v_norm = v / v_sum if v_sum > 0 else v
                u_norm = u / u_sum if u_sum > 0 else u
                
                # Sprawdzamy ograniczenia wagowe na znormalizowanych wagach
                is_feasible = True
                if weight_constraints:
                    w = np.concatenate([v_norm, u_norm])
                    for a_l in weight_constraints:
                        if np.dot(a_l, w) > 1e-7:
                            is_feasible = False
                            break
                            
                if is_feasible:
                    ratios = {}
                    for _, row in df.iterrows():
                        weighted_output = sum(u_norm[r] * row[outputs[r]] for r in range(len(outputs)))
                        weighted_input = sum(v_norm[i] * row[inputs[i]] for i in range(len(inputs)))
                        ratios[orig_id_type(row['ID'])] = weighted_output / weighted_input if weighted_input > 1e-12 else 0.0
                    
                    smaa_results.append({
                        'weights_v': v_norm,
                        'weights_u': u_norm,
                        'ratios': ratios,
                        'winner': target
                    })
                    winners_in_smaa.add(target)
                    print(f"  [SMAA sampling Augmentation] Dodano optymalną próbkę wag dla jednostki: {target}")

    print(f"====================================================")
    print(f"RAPORT WYJAŚNIEŃ DEA DLA DATASETU: {dataset_name.upper()}")
    print(f"====================================================\n")

    # 2. Generowanie reguł globalnych (P1)
    rules_text = xplainer.train_p1(max_depth=3)
    print("### P1: Globalne reguły rozdzielające ###")
    print(rules_text)

    # Przygotowanie DataFrame z wag i zwycięzcy dla sub-explainerów wagowych
    weight_data = []
    for s in smaa_results:
        row = {}
        for i, name in enumerate(inputs):
            row[f"v_{name}"] = s['weights_v'][i]
        for r, name in enumerate(outputs):
            row[f"u_{name}"] = s['weights_u'][r]
        row['best_dmu'] = s['winner']
        weight_data.append(row)
    weight_samples_df = pd.DataFrame(weight_data)
    
    # Wywołanie W2 (mapa przestrzeni wag)
    xplainer.explain_weight_preferences(weight_samples_df)

    # U5 globalnie
    bench_freq = xplainer.unit_explainer.explain_u5(smaa_results)
    print(f"\n[U5] Częstotliwość bycia benchmarkiem: {bench_freq}")

    print("\n--- ANALIZA SZCZEGÓŁOWA DLA KAŻDEJ JEDNOSTKI ---\n")

    for target in df['ID']:
        is_eff = df.loc[df['ID'] == target, 'is_efficient'].values[0]
        status_str = "EFEKTYWNA" if is_eff == 1 else "NIEFEKTYWNA"

        print(f"--- JEDNOSTKA: {target} [{status_str}] ---")

        # U1 - U5
        if is_eff == 0:
            # P2: Wyświetlanie projekcji i rówieśników
            xplainer.visualize_p2(target)

            witnesses = xplainer.explain_unit(target)
            print(f"  [U1] Zbiory świadków nieefektywności: {witnesses}")

            u2_removal_set = xplainer.explain_unit_removal(target)
            print(f" [U2] Zbiory do usunięcia aby {target} został efektywny to: {u2_removal_set}")

            u3_rivals = xplainer.unit_explainer.explain_u3(target, smaa_results)
            print(f"  [U3] Minimalny zbiór rywali (scenariusze): {u3_rivals}")

            u4_rivals = xplainer.unit_explainer.explain_u4(target, smaa_results)
            print(f"  [U4] Rywale uniwersalni: {u4_rivals['universal']}")
            print(f"  [U4] Rywale obligatoryjni: {u4_rivals['mandatory']}")
            
            # C1 i C2: Wyjaśnienia oparte na ograniczeniach wagowych
            # Prerequisite check (§ Constraint-based explanations):
            # C1/C2 sensowne tylko gdy ρ*(∅)=1 ale ρ*(C)<1 (ograniczenia "tworzą" nieefektywność)
            if weight_constraints:
                eff_no_constraints, _, _ = solve_ccr_multiplier(df, inputs, outputs, target, weight_constraints=None)
                eff_with_constraints, _, _ = solve_ccr_multiplier(df, inputs, outputs, target, weight_constraints=weight_constraints)
                
                if eff_no_constraints is not None and eff_no_constraints >= 1 - 1e-4 and \
                   eff_with_constraints is not None and eff_with_constraints < 1 - 1e-4:
                    # Ograniczenia odpowiadają za nieefektywność — C1/C2 mają sens
                    c1_removals = xplainer.explain_constraints_removal(target)
                    print(f"  [C1] Zbiory ograniczeń do usunięcia aby {target} został efektywny: {c1_removals}")
                    c2_core = xplainer.explain_core_constraints(target)
                    print(f"  [C2] Rdzeń ograniczeń wymuszający nieefektywność {target}: {c2_core}")
                else:
                    if eff_no_constraints is not None and eff_no_constraints < 1 - 1e-4:
                        print(f"  [C1/C2] Pominięto — {target} jest nieefektywna już bez ograniczeń (ρ*(∅)={eff_no_constraints:.4f}), nieefektywność wynika z wyników.")
                    else:
                        print(f"  [C1/C2] Pominięto — warunki wstępne niespełnione (ρ*(∅)={eff_no_constraints}, ρ*(C)={eff_with_constraints}).")
        else:
            print(f"  [U1-U5] Pomijane ze względu na efektywność")
            # W1: Kiedy dana efektywna jednostka jest liderem
            xplainer.explain_weight_preferences(weight_samples_df, target_id=target)

        # F1 i F2
        if is_eff == 1:
            # F1
            sufficient_sets = xplainer.explain_factors(target)
            print(f"  [F1] Minimalne zestawy cech dla sukcesu: {sufficient_sets}")

            # F2
            f2_res = xplainer.explain_critical_factors(target)
            print(f"  [F2] Cechy krytyczne (ich brak niszczy efektywność): {f2_res}")
        else:
            print(f"  [F1/F2] Analiza czynników pominięta (jednostka nieefektywna).")

        print("-" * 30)


def run_analysis():
    # 1. Lotniska bez ograniczeń
    run_analysis_for_dataset(
        data_loader.get_airport_data,
        "Lotniska (bez ograniczeń)",
        # "DEA_inputs/airports/airports_input_weights.csv",
        # "DEA_inputs/airports/airports_output_weights.csv",
        None, # aby samemu wygenerować więcej próbek
        None, # aby samemu wygenerować więcej próbek
        efficient_ids=['WAW', 'KRK', 'WRO', 'GDN', 'BZG']
    )
    
    print("\n" + "="*80 + "\n")
    
    # 2. Lotniska z ograniczeniami custom
    # Ograniczenia wagowe z PolishAirportsExample.java (addWeightConstraints):
    # Format: a_l^T * [v_i1, v_i2, v_i3, v_i4, u_o1, u_o2] <= 0
    airport_custom_constraints = [
        [-1, 0, 3, 0, 0, 0],   # w(i1) >= 3*w(i3)  → -v_i1 + 3*v_i3 <= 0
        [-1, 0, 0, 5, 0, 0],   # w(i1) >= 5*w(i4)  → -v_i1 + 5*v_i4 <= 0
        [0, -1, 2, 0, 0, 0],   # w(i2) >= 2*w(i3)  → -v_i2 + 2*v_i3 <= 0
        [0, -1, 0, 5, 0, 0],   # w(i2) >= 5*w(i4)  → -v_i2 + 5*v_i4 <= 0
        [0, 0, 0, 0, -1, 5],   # w(o1) >= 5*w(o2)  → -u_o1 + 5*u_o2 <= 0
    ]
    run_analysis_for_dataset(
        data_loader.get_airport_data,
        "Lotniska (z ograniczeniami custom)",
        # "DEA_inputs/airports/airports_input_weights_custom.csv",
        # "DEA_inputs/airports/airports_output_weights_custom.csv",
        None, # aby samemu wygenerować więcej próbek
        None, # aby samemu wygenerować więcej próbek
        efficient_ids=['WAW', 'GDN'],
        weight_constraints=airport_custom_constraints
    )
    
    print("\n" + "="*80 + "\n")
    
    # 3. Roboty
    run_analysis_for_dataset(
        data_loader.get_robot_data,
        "Roboty",
        # "DEA_inputs/robots/robots_input_weights.csv",
        # "DEA_inputs/robots/robots_output_weights.csv",
        None, # aby samemu wygenerować więcej próbek
        None, # aby samemu wygenerować więcej próbek
        efficient_ids=[1, 4, 7, 10, 13, 14, 19, 20, 27]
    )

    print("\n" + "="*80 + "\n")

    # 4. Running Example z dea-explanations.tex (Tabela 1)
    running_example_constraints = [
        [-1, 1, 0, 0],  # v_2 - v_1 <= 0
        [1, -3, 0, 0],  # v_1 - 3*v_2 <= 0
        [0, 0, -1, 1.4], # 1.4*u_2 - u_1 <= 0
        [0, 0, 1, -3]   # u_1 - 3*u_2 <= 0
    ]
    run_analysis_for_dataset(
        data_loader.get_running_example_data,
        "Running Example",
        efficient_ids=['A', 'B', 'F', 'G'],
        weight_constraints=running_example_constraints
    )

    # 5. Szpitale  # pytanie czy to ma sens jak tam był inny model
    # run_analysis_for_dataset(
    #     data_loader.get_hospital_data,
    #     "Szpitale",
    #     "DEA_inputs/hospitals/hospitals_input_weights.csv",
    #     "DEA_inputs/hospitals/hospitals_output_weights.csv",
    #     efficient_ids=[10, 17, 20, 25, 27]
    # )

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    run_analysis()
