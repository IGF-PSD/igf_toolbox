import numpy as np
import pandas as pd

from igf_toolbox.stats_des.base2 import StatDesGroupBy

# Simuler la classe StatDesGroupBy et les fonctions auxiliaires nécessaires
# (Dans un vrai contexte, ces imports viendraient des modules appropriés)

# Génération d'un jeu de données synthétique
np.random.seed(42)

# Créer un dataset d'employés dans différentes entreprises
n_employees = 1000

# Variables de base
data = pd.DataFrame({
    'employee_id': [f'EMP_{i:04d}' for i in range(n_employees)],
    'company_id': np.random.choice([f'COMP_{i:02d}' for i in range(10)], n_employees),
    'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Operations'], n_employees),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_employees),
    'gender': np.random.choice(['M', 'F'], n_employees, p=[0.6, 0.4]),
    'age': np.random.randint(22, 65, n_employees),
    'experience_years': np.random.randint(0, 30, n_employees),
    'salary': np.random.normal(50000, 20000, n_employees).clip(20000, 150000).astype(int),
    'bonus': np.random.exponential(5000, n_employees).clip(0, 50000).astype(int),
    'hours_worked': np.random.normal(40, 5, n_employees).clip(20, 60),
    'satisfaction_score': np.random.uniform(1, 10, n_employees),
    'is_manager': np.random.choice([True, False], n_employees, p=[0.15, 0.85]),
    'has_certification': np.random.choice([True, False], n_employees, p=[0.3, 0.7]),
    'weight': np.random.uniform(0.5, 2.0, n_employees)  # Poids pour les calculs pondérés
})

# Ajouter quelques valeurs manquantes pour tester la robustesse
data.loc[np.random.choice(data.index, 50), 'bonus'] = np.nan
data.loc[np.random.choice(data.index, 30), 'satisfaction_score'] = np.nan

print("=== JEU DE DONNÉES SYNTHÉTIQUE ===")
print(f"Nombre d'employés: {len(data)}")
print(f"Nombre d'entreprises: {data['company_id'].nunique()}")
print(f"Départements: {data['department'].unique()}")
print(f"Régions: {data['region'].unique()}")
print("\nAperçu des données:")
print(data.head(10))
print("\nStatistiques descriptives:")
print(data.describe())

# ========================================
# EXEMPLES D'UTILISATION DE StatDesGroupBy
# ========================================

print("\n\n=== EXEMPLE 1: Opérations simples sans groupby ===")
# Calcul de statistiques globales
stat_des = StatDesGroupBy(
    data_source=data,
    list_var_groupby=[],
    list_var_of_interest=['salary', 'bonus', 'hours_worked'],
    var_count='employee_id',
    var_weights=None
)

# Liste d'opérations simples
operations = ['sum', 'mean', 'count', 'nunique']
result = stat_des.iterate_without_total(operations)
print("\nRésultats (opérations simples sans groupby):")
print(result)

print("\n\n=== EXEMPLE 2: Opérations avec un niveau de groupby ===")
# Statistiques par département
stat_des_dept = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['department'],
    list_var_of_interest=['salary', 'bonus', 'hours_worked'],
    var_count='employee_id',
    var_weights=None
)

# Dictionnaire d'opérations
operations_dict = {
    'sum': ['salary', 'bonus'],
    'mean': ['salary', 'hours_worked'],
    'count_effectif': []  # Utilisera var_count
}
result_dept = stat_des_dept.iterate_with_total(operations_dict)
print("\nRésultats par département (avec totaux):")
print(result_dept)

print("\n\n=== EXEMPLE 3: Opérations pondérées ===")
# Statistiques pondérées par région
stat_des_weighted = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['region'],
    list_var_of_interest=['salary', 'bonus', 'satisfaction_score'],
    var_count='employee_id',
    var_weights='weight'
)

# Opérations incluant des calculs pondérés
weighted_operations = {
    'sum': ['salary', 'weight'],  # La somme des poids sera calculée séparément
    'mean': ['salary', 'satisfaction_score'],  # Moyennes pondérées
    'median': ['salary']  # Médiane pondérée
}
result_weighted = stat_des_weighted.iterate_with_total(weighted_operations)
print("\nRésultats pondérés par région:")
print(result_weighted)

print("\n\n=== EXEMPLE 4: Opérations avec plusieurs niveaux de groupby ===")
# Statistiques par région et département
stat_des_multi = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['region', 'department'],
    list_var_of_interest=['salary', 'bonus', 'is_manager'],
    var_count=['employee_id', 'company_id'],  # Compter employés ET entreprises
    var_weights=None
)

# Opérations complexes
multi_operations = {
    'mean': ['salary'],
    'count_effectif': [],  # Comptera employee_id et company_id
    'any': ['is_manager'],  # Y a-t-il au moins un manager ?
    'majority': ['gender']  # Genre majoritaire
}
result_multi = stat_des_multi.iterate_with_total(multi_operations)
print("\nRésultats par région et département (avec sous-totaux):")
print(result_multi.head(20))  # Afficher les premières lignes

print("\n\n=== EXEMPLE 5: Opérations avec quantiles ===")
# Calcul de quantiles
stat_des_quantiles = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['department'],
    list_var_of_interest=['salary', 'age', 'experience_years'],
    var_count=None,
    var_weights='weight'
)

# Opérations avec quantiles
quantile_operations = [
    'mean',
    ('quantile', {'q': 0.25}),
    'median',  # équivalent à quantile 0.5
    ('quantile', {'q': 0.75})
]
result_quantiles = stat_des_quantiles.iterate_without_total(quantile_operations)
print("\nQuantiles pondérés par département:")
print(result_quantiles)

print("\n\n=== EXEMPLE 6: Opérations spéciales - Proportions ===")
# Calculer des proportions
# Créer une variable de référence (total des compensations)
data['total_compensation'] = data['salary'] + data['bonus'].fillna(0)

stat_des_prop = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['region'],
    list_var_of_interest=['salary', 'bonus', 'total_compensation'],
    var_count=None,
    var_weights='weight'
)

# Opérations de proportion
# Note: Les tuples avec paramètres doivent avoir un format hashable
prop_operations = {
    'sum': ['salary', 'bonus', 'total_compensation'],
    ('prop', tuple([('var_ref', 'total_compensation')])): ['salary', 'bonus']
}
result_prop = stat_des_prop.iterate_with_total(prop_operations)
print("\nProportions du salaire et bonus par rapport à la compensation totale:")
print(result_prop)

print("\n\n=== EXEMPLE 7: Opérations spéciales - Seuils ===")
# Analyser la proportion d'employés sous un seuil d'âge
stat_des_threshold = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['department'],
    list_var_of_interest=['employee_id', 'company_id'],
    var_count=None,
    var_weights=None
)

# Opérations avec seuils
threshold_operations = [
    'nunique',
    ('inf_threshold', {'var_threshold': 'age', 'threshold': 30})
]
result_threshold = stat_des_threshold.iterate_without_total(threshold_operations)
print("\nProportion d'employés de moins de 30 ans par département:")
print(result_threshold)

print("\n\n=== EXEMPLE 8: Opérations max/sum ===")
# Analyser la concentration (part du plus gros contributeur)
stat_des_concentration = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['region'],
    list_var_of_interest=['salary', 'bonus'],
    var_count='company_id',
    var_weights='weight'
)

# Opérations de concentration
concentration_operations = {
    'sum': ['salary', 'bonus'],
    'max_sum_effectif': ['salary', 'bonus']
}
result_concentration = stat_des_concentration.iterate_with_total(concentration_operations)
print("\nConcentration des salaires et bonus par région:")
print(result_concentration)

print("\n\n=== EXEMPLE 9: Cas complet avec toutes les fonctionnalités ===")
# Exemple combinant plusieurs types d'opérations
stat_des_complete = StatDesGroupBy(
    data_source=data,
    list_var_groupby=['region', 'gender'],
    list_var_of_interest=['salary', 'bonus', 'hours_worked', 'satisfaction_score', 'has_certification'],
    var_count=['employee_id', 'company_id'],
    var_weights='weight',
    dropna=True  # Supprimer les valeurs manquantes
)

# Dictionnaire complet d'opérations
# Note: Les tuples avec paramètres doivent avoir un format hashable
complete_operations = {
    'count': ['salary'],  # Nombre de non-NA
    'nunique': ['company_id'],  # Nombre d'entreprises uniques
    'sum': ['salary', 'weight'],
    'mean': ['salary', 'satisfaction_score'],
    'median': ['salary'],
    ('quantile', tuple([('q', 0.9)])): ['salary'],
    'any': ['has_certification'],
    'all': ['has_certification'],
    'majority': ['department'],
    'count_effectif': [],
    ('prop', tuple([('var_ref', 'hours_worked')])): ['salary'],
    'max_sum_effectif': ['bonus']
}

result_complete = stat_des_complete.iterate_with_total(complete_operations)
print("\nRésultats complets par région et genre:")
print(result_complete.head(15))
print(f"\nNombre total de lignes (avec totaux et sous-totaux): {len(result_complete)}")

print("\n\n=== EXEMPLE 10: Comparaison avec/sans totaux ===")
# Même analyse sans les totaux
result_no_total = stat_des_complete.iterate_without_total(complete_operations)
print(f"\nNombre de lignes sans totaux: {len(result_no_total)}")
print("Premières lignes sans totaux:")
print(result_no_total.head())

# Vérifier que les totaux sont bien calculés
print("\n\nVérification des totaux:")
print("Total général (dernière ligne avec totaux):")
print(result_complete.loc[('Total', 'Total')])

print("\n\n=== RÉSUMÉ DES CAS D'USAGE COUVERTS ===")
print("""
1. ✓ Opérations simples sans groupby
2. ✓ Opérations avec un niveau de groupby
3. ✓ Opérations pondérées (sum, mean, median, quantiles)
4. ✓ Opérations avec plusieurs niveaux de groupby et sous-totaux
5. ✓ Utilisation de var_count (string ou liste)
6. ✓ Format dictionnaire et liste pour iterable_operations
7. ✓ Opérations spéciales (majority, prop, inf_threshold, max_sum_effectif)
8. ✓ Gestion des totaux et sous-totaux
9. ✓ Option dropna pour gérer les valeurs manquantes
10. ✓ Combinaison de toutes les fonctionnalités
""")