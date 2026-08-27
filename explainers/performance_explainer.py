from pulp import *
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd


class PerformanceExplainer:
    def __init__(self, df, inputs, outputs):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.features = inputs + outputs

    def explain_p1(self, max_depth=3):
        """
        P1: Globalne reguły rozdzielające jednostki
        """
        X = self.df[self.features].values
        y = self.df['is_efficient'].values

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y)

        rules = export_text(clf, feature_names=self.features)
        return clf, rules

    def explain_p2(self, target_id):
        """
        P2: Dokładna projekcja i rówieśnicy (Peers) - model CCR Envelopment (Eq. 132).
        Orientacja na nakłady (input-oriented).
        """
        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]

        prob = LpProblem(f"P2_Projection_{target_id}", LpMinimize)

        # Zmienne: theta (skalar efektywności), lambda (wagi rówieśników), slacks
        theta = LpVariable("theta", lowBound=0)
        lambdas = LpVariable.dicts("lambda", self.df.index, lowBound=0)
        s_minus = LpVariable.dicts("s_minus", self.inputs, lowBound=0)
        s_plus = LpVariable.dicts("s_plus", self.outputs, lowBound=0)

        # Cel: Minimalizacja theta (Eq. 132)
        prob += theta

        # Ograniczenia dla wejść (inputs)
        for m in self.inputs:
            prob += lpSum([lambdas[k] * self.df.loc[k, m] for k in self.df.index]) + s_minus[m] == theta * target_dmu[m]

        # Ograniczenia dla wyjść (outputs)
        for n in self.outputs:
            prob += lpSum([lambdas[k] * self.df.loc[k, n] for k in self.df.index]) - s_plus[n] == target_dmu[n]

        status = prob.solve(PULP_CBC_CMD(msg=0))

        if LpStatus[status] == 'Optimal':
            # Rówieśnicy to jednostki, dla których lambda > 0 (Eq. 133)
            peers = {self.df.loc[k, 'ID']: round(lambdas[k].varValue, 4)
                     for k in self.df.index if lambdas[k].varValue > 1e-5}

            # Slacks (wąskie gardła)
            slacks = {
                'inputs': {m: round(s_minus[m].varValue, 4) for m in self.inputs if s_minus[m].varValue > 1e-5},
                'outputs': {n: round(s_plus[n].varValue, 4) for n in self.outputs if s_plus[n].varValue > 1e-5}
            }

            return {
                'theta': round(theta.varValue, 4),
                'peers': peers,
                'slacks': slacks
            }
        return None

    def visualize_p2(self, target_id, projection_results):
        """
        Wizualizacja projekcji P2 dla wybranej jednostki.
        """
        if not projection_results:
            print(f"Brak wyników projekcji dla {target_id}.")
            return

        target_dmu = self.df[self.df['ID'] == target_id].iloc[0]
        theta = projection_results['theta']
        peers = projection_results['peers']
        slacks = projection_results['slacks']

        print(f"\n--- [P2] Raport Projekcji i Rówieśników dla {target_id} ---")
        print(f"  Optymalny współczynnik efektywności (theta): {theta:.4f}")
        
        # Wyświetlamy rówieśników (peers)
        if peers:
            peer_strs = [f"{peer_id} (lambda={intensity:.4f})" for peer_id, intensity in peers.items()]
            print(f"  Jednostki rówieśnicze (Peers): {', '.join(peer_strs)}")
        else:
            print("  Brak jednostek rówieśniczych.")

        # Tabela porównawcza dla wejść (inputs)
        print("\n  ANALIZA CECH WEJŚCIOWYCH (INPUTS):")
        print(f"  {'Cecha':<15} | {'Oryginalna':<12} | {'Docelowa':<12} | {'Slack':<12} | {'Redukcja %':<12}")
        print("  " + "-" * 73)
        for m in self.inputs:
            orig = target_dmu[m]
            slack = slacks['inputs'].get(m, 0.0)
            target_val = theta * orig - slack
            reduction = orig - target_val
            pct_reduction = (reduction / orig * 100.0) if orig > 0 else 0.0
            print(f"  {m:<15} | {orig:<12.4f} | {target_val:<12.4f} | {slack:<12.4f} | {pct_reduction:<11.2f}%")

        # Tabela porównawcza dla wyjść (outputs)
        print("\n  ANALIZA CECH WYJŚCIOWYCH (OUTPUTS):")
        print(f"  {'Cecha':<15} | {'Oryginalna':<12} | {'Docelowa':<12} | {'Slack':<12} | {'Wzrost %':<12}")
        print("  " + "-" * 73)
        for n in self.outputs:
            orig = target_dmu[n]
            slack = slacks['outputs'].get(n, 0.0)
            target_val = orig + slack
            pct_increase = (slack / orig * 100.0) if orig > 0 else 0.0
            print(f"  {n:<15} | {orig:<12.4f} | {target_val:<12.4f} | {slack:<12.4f} | {pct_increase:<11.2f}%")
        print("\n" + "-" * 30 + "\n")

