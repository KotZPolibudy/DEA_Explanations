import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


class WeightExplainer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weight_names = [f"v_{i}" for i in inputs] + [f"u_{r}" for r in outputs]

    def explain_w1(self, weight_samples, target_id, max_depth=3):
        """
        W1: Reguły binarne dla jednej jednostki.
        weight_samples: DataFrame z kolumnami wag i kolumną 'best_dmu'
        """
        X = weight_samples[self.weight_names]
        # Etykieta: 1 jeśli target_id był najlepszy w danej próbie
        y = (weight_samples['best_dmu'] == target_id).astype(int)

        if y.sum() == 0:
            return None, f"Jednostka {target_id} nigdy nie jest liderem w podanych próbkach wag."

        clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced', random_state=42)
        clf.fit(X, y)

        rules = export_text(clf, feature_names=self.weight_names, decimals=3)
        return clf, rules

    def explain_w2(self, weight_samples, max_depth=4):
        """
        W2: Mapa całej przestrzeni wag (Multi-class).
        Pokazuje regiony 'panowania' poszczególnych benchmarków.
        """
        X = weight_samples[self.weight_names]
        y = weight_samples['best_dmu']

        clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced', random_state=42)
        clf.fit(X, y)

        rules = export_text(clf, feature_names=self.weight_names, decimals=3)
        return clf, rules