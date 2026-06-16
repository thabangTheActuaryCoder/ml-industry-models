"""Generate synthetic claim fraud detection data for the insurance industry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from shared.config import RANDOM_SEED, INSURANCE_DIR, INCIDENT_TYPES, SA_PROVINCES
from shared.data_generation import BaseSyntheticDataGenerator


class ClaimFraudDataGenerator(BaseSyntheticDataGenerator):
    """Generates synthetic insurance claim data with ~8% fraud rate (imbalanced)."""

    def _generate_features(self):
        n = self.n_samples
        rng = self.rng

        df = pd.DataFrame({
            "claim_amount": np.round(rng.lognormal(mean=9.5, sigma=1.0, size=n), 2),
            "policy_tenure_months": rng.integers(1, 240, size=n),
            "customer_age": rng.integers(18, 75, size=n),
            "num_prior_claims": rng.poisson(lam=1.2, size=n),
            "premium_amount": np.round(rng.lognormal(mean=7.5, sigma=0.5, size=n), 2),
            "days_to_report": rng.integers(0, 90, size=n),
            "witness_present": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
            "police_report_filed": rng.choice([0, 1], size=n, p=[0.5, 0.5]),
            "incident_type": rng.choice(INCIDENT_TYPES, size=n),
            "vehicle_age_years": rng.integers(0, 25, size=n),
            "province": rng.choice(SA_PROVINCES, size=n),
        })

        return df

    def _generate_target(self, df):
        """Generate fraud labels (~8% fraud rate) based on suspicious patterns."""
        rng = self.rng

        risk_score = (
            0.00001 * df["claim_amount"]
            + 0.15 * df["num_prior_claims"]
            + 0.02 * df["days_to_report"]
            - 0.3 * df["witness_present"]
            - 0.4 * df["police_report_filed"]
            - 0.005 * df["policy_tenure_months"]
            + 0.03 * df["vehicle_age_years"]
        )

        # Calibrate for ~8% fraud rate
        threshold = np.percentile(risk_score, 92)
        prob = 1 / (1 + np.exp(-5.0 * (risk_score - threshold)))
        noise = rng.normal(0, 0.01, size=len(df))
        prob = np.clip(prob + noise, 0.001, 0.85)

        return (rng.random(len(df)) < prob).astype(int)


if __name__ == "__main__":
    output_path = INSURANCE_DIR / "data" / "claim_data.csv"
    generator = ClaimFraudDataGenerator(n_samples=8_000)
    df = generator.generate_and_save(output_path)
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
