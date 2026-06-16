"""Generate synthetic customer churn data for the retail industry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from shared.config import RANDOM_SEED, RETAIL_DIR, SA_PROVINCES
from shared.data_generation import BaseSyntheticDataGenerator


class CustomerChurnDataGenerator(BaseSyntheticDataGenerator):
    """Generates synthetic retail customer churn data with ~20% churn rate."""

    def _generate_features(self):
        n = self.n_samples
        rng = self.rng

        membership_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        channels = ["Online", "In-store", "Both"]

        df = pd.DataFrame({
            "monthly_spend_zar": np.round(rng.lognormal(mean=7.0, sigma=0.8, size=n), 2),
            "days_since_last_purchase": rng.integers(0, 365, size=n),
            "num_support_tickets": rng.poisson(lam=2.0, size=n),
            "loyalty_points": rng.integers(0, 50_000, size=n),
            "account_age_months": rng.integers(1, 120, size=n),
            "num_returns_last_year": rng.poisson(lam=1.0, size=n),
            "avg_order_value_zar": np.round(rng.lognormal(mean=6.0, sigma=0.6, size=n), 2),
            "num_orders_last_6m": rng.integers(0, 30, size=n),
            "discount_usage_rate": np.round(rng.beta(2, 5, size=n), 3),
            "membership_tier": rng.choice(membership_tiers, size=n),
            "preferred_channel": rng.choice(channels, size=n),
            "province": rng.choice(SA_PROVINCES, size=n),
        })

        return df

    def _generate_target(self, df):
        """Generate churn labels (~20% churn rate)."""
        rng = self.rng

        risk_score = (
            0.005 * df["days_since_last_purchase"]
            + 0.15 * df["num_support_tickets"]
            - 0.00001 * df["monthly_spend_zar"]
            - 0.00002 * df["loyalty_points"]
            - 0.005 * df["account_age_months"]
            + 0.1 * df["num_returns_last_year"]
            - 0.05 * df["num_orders_last_6m"]
        )

        threshold = np.percentile(risk_score, 80)
        prob = 1 / (1 + np.exp(-3.0 * (risk_score - threshold)))
        noise = rng.normal(0, 0.02, size=len(df))
        prob = np.clip(prob + noise, 0.005, 0.90)

        return (rng.random(len(df)) < prob).astype(int)


if __name__ == "__main__":
    output_path = RETAIL_DIR / "data" / "customer_data.csv"
    generator = CustomerChurnDataGenerator(n_samples=12_000)
    df = generator.generate_and_save(output_path)
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
