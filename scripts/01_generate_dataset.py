"""
Étape 1 — Génération et validation d'un dataset DVRP.

    python scripts/01_generate_dataset.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generator import generate_dataset, save_dataset, load_dataset
from data.validation import validate_dataset, validate_instance, print_validation_report

OUT = "outputs/datasets"

def main(n_train=1000, n_test=256, n_customers=20, spatial_method="uniform"):
    os.makedirs(OUT, exist_ok=True)

    print(f"Génération : {n_train} instances d'entraînement, {n_test} de test "
          f"({n_customers} clients, spatial={spatial_method})\n")

    train = generate_dataset(n_train, base_seed=0,
                             n_customers=n_customers, spatial_method=spatial_method)
    test = generate_dataset(n_test, base_seed=20_000_000,
                            n_customers=n_customers, spatial_method=spatial_method)

    print("Validation du jeu d'entraînement :")
    validate_dataset(train)
    print("\nValidation du jeu de test :")
    validate_dataset(test)

    p1 = save_dataset(train, f"{OUT}/dvrp_{n_customers}_train.npz")
    p2 = save_dataset(test, f"{OUT}/dvrp_{n_customers}_test.npz")
    print(f"\nSauvegardé : {p1}\n            {p2}")

    print("\nExemple d'instance :")
    print(train[0].summary())
    return train, test

if __name__ == "__main__":
    main()
