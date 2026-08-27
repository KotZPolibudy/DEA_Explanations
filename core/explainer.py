import dtreeviz
from sklearn.tree import DecisionTreeClassifier, export_text
from explainers.performance_explainer import PerformanceExplainer
from explainers.unit_explainer import UnitExplainer
from explainers.factor_explainer import FactorExplainer
from explainers.weight_explainer import WeightExplainer
from explainers.constraint_explainer import ConstraintExplainer


class DEAXplainer:
    def __init__(self, df, inputs, outputs, dataset_name="DEA Dataset", weight_constraints=None):
        self.df = df
        self.dataset_name = dataset_name
        self.inputs = inputs
        self.outputs = outputs
        self.features = inputs + outputs
        self.weight_constraints = weight_constraints

        # Inicjalizacja sub-explainerów
        self.performance_explainer = PerformanceExplainer(df, inputs, outputs)
        self.unit_explainer = UnitExplainer(df, inputs, outputs, weight_constraints)
        self.factor_explainer = FactorExplainer(df, inputs, outputs, weight_constraints)
        self.weight_explainer = WeightExplainer(inputs, outputs)
        self.constraint_explainer = ConstraintExplainer(df, inputs, outputs, weight_constraints)
        self.clf = None

    def train_p1(self, max_depth=3):
        """Metoda P1: Reguły globalne."""
        clf, rules = self.performance_explainer.explain_p1(max_depth)
        self.clf = clf  # dla wizualizacji dtreeviz
        return rules

    def visualize_p1(self):
        """Generuje wizualizację dtreeviz dla metody P1."""
        if self.clf is None:
            raise ValueError("Model nie jest wytrenowany! Wywołaj najpierw train_p1().")

        # Konfiguracja dtreeviz
        viz = dtreeviz.model(self.clf,
                             X_train=self.df[self.features],
                             y_train=self.df['is_efficient'],
                             target_name="Efektywność",
                             feature_names=self.features,
                             class_names=["Nieefektywne", "Efektywne"])

        return viz.view()

    def explain_projection(self, target_id):
        """Metoda P2: Wyjaśnienie poprzez rówieśników i projekcję."""
        # P2 ma największy sens dla jednostek nieefektywnych
        return self.performance_explainer.explain_p2(target_id)

    def visualize_p2(self, target_id):
        """Wizualizacja projekcji P2 dla wybranenej jednostki."""
        results = self.performance_explainer.explain_p2(target_id)
        if results:
            self.performance_explainer.visualize_p2(target_id, results)
        else:
            print(f"Brak wyników projekcji dla {target_id}.")

    def explain_factors(self, target_id):
        """Wywołuje metodę F1 dla jednostki."""
        # F1/F2 ma sens głównie dla jednostek efektywnych
        status = self.df[self.df['ID'] == target_id]['is_efficient'].values[0]
        if status == 0:
            return f"Jednostka {target_id} nie jest efektywna. F1 dotyczy redukcji cech dla liderów."

        sufficient_factors = self.factor_explainer.explain_f1(target_id)
        return sufficient_factors

    def explain_critical_factors(self, target_id):
        """Wywołuje metodę F2: szuka krytycznych cech dla lidera."""
        status = self.df[self.df['ID'] == target_id]['is_efficient'].values[0]
        if status == 0:
            return f"Jednostka {target_id} już jest nieefektywna."

        critical_factors = self.factor_explainer.explain_f2(target_id)
        return critical_factors

    def explain_weight_preferences(self, weight_samples_df, target_id=None):
        """
        Wywołuje W1 (jeśli podano target_id) lub W2 (ogólna mapa).
        """
        if target_id:
            clf, rules = self.weight_explainer.explain_w1(weight_samples_df, target_id)
            print(f"\n### W1: Kiedy {target_id} jest najlepszy? ###\n{rules}")
        else:
            clf, rules = self.weight_explainer.explain_w2(weight_samples_df)
            print(f"\n### W2: Mapa liderów przestrzeni wag ###\n{rules}")
        return clf

    def explain_unit(self, target_id):
        """Deleguje zadanie do UnitExplainer (Metoda U1)."""
        return self.unit_explainer.explain_u1(target_id)

    def explain_unit_removal(self, target_id):
        """Wywołuje metodę U2 dla wybranej jednostki."""
        # U2 ma sens tylko dla jednostek nieefektywnych
        status = self.df[self.df['ID'] == target_id]['is_efficient'].values[0]
        if status == 1:
            return []  # Jednostka już jest efektywna

        return self.unit_explainer.explain_u2(target_id)

    def explain_constraints_removal(self, target_id):
        """Wywołuje metodę C1 dla wybranej jednostki (usunięcie ograniczeń)."""
        return self.constraint_explainer.explain_c1(target_id)

    def explain_core_constraints(self, target_id):
        """Wywołuje metodę C2 dla wybranej jednostki (rdzeń ograniczeń)."""
        return self.constraint_explainer.explain_c2(target_id)