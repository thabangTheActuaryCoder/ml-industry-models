"""Generate synthetic credit scoring data for the banking industry."""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from shared.config import RANDOM_SEED, BANKING_DIR, EDUCATION_LEVELS, EMPLOYMENT_TYPES, SA_PROVINCES
from shared.data_generation import BaseSyntheticDataGenerator


class CreditScoringDataGenerator(BaseSyntheticDataGenerator):
    """Generates synthetic credit data with ~15% default rate."""

    def _generate_features(self):
        n = self.n_samples
        rng = self.rng

        df = pd.DataFrame({
            "age": rng.integers(21, 65, size=n),
            "annual_income": rng.lognormal(mean=12.0, sigma=0.6, size=n).astype(int),
            "employment_years": rng.integers(0, 30, size=n),
            "loan_amount": rng.lognormal(mean=11.5, sigma=0.8, size=n).astype(int),
            "credit_score": rng.integers(300, 850, size=n),
            "num_late_payments": rng.poisson(lam=1.5, size=n),
            "debt_to_income_ratio": np.round(rng.uniform(0.05, 0.8, size=n), 3),
            "num_open_accounts": rng.integers(1, 15, size=n),
            "months_since_last_delinquency": rng.integers(0, 120, size=n),
            "education_level": rng.choice(EDUCATION_LEVELS, size=n),
            "employment_type": rng.choice(EMPLOYMENT_TYPES, size=n),
            "province": rng.choice(SA_PROVINCES, size=n),
        })

        return df

    def _generate_target(self, df):
        """Generate default labels (~15% default rate) based on realistic risk signals."""
        rng = self.rng

        # Build a risk score from features
        risk_score = (
            -0.02 * df["credit_score"]
            + 0.3 * df["num_late_payments"]
            + 2.0 * df["debt_to_income_ratio"]
            - 0.00001 * df["annual_income"]
            + 0.00001 * df["loan_amount"]
            - 0.02 * df["employment_years"]
            - 0.01 * df["age"]
        )

        # Normalise to probability via logistic function and calibrate for ~15% positive rate
        prob = 1 / (1 + np.exp(-(risk_score - np.percentile(risk_score, 85))))
        noise = rng.normal(0, 0.05, size=len(df))
        prob = np.clip(prob + noise, 0.01, 0.99)

        return (rng.random(len(df)) < prob).astype(int)


if __name__ == "__main__":
    output_path = BANKING_DIR / "data" / "credit_data.csv"
    generator = CreditScoringDataGenerator(n_samples=10_000)
    df = generator.generate_and_save(output_path)
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
