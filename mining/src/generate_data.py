"""Generate synthetic equipment failure data for the mining industry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from shared.config import RANDOM_SEED, MINING_DIR, SA_MINING_EQUIPMENT, SA_PROVINCES
from shared.data_generation import BaseSyntheticDataGenerator


class EquipmentFailureDataGenerator(BaseSyntheticDataGenerator):
    """Generates synthetic mining equipment data with ~10% failure rate."""

    def _generate_features(self):
        n = self.n_samples
        rng = self.rng

        mine_types = ["Open Pit", "Underground", "Alluvial"]
        shift_types = ["Day", "Night", "Extended"]

        df = pd.DataFrame({
            "temperature_celsius": np.round(rng.normal(75, 15, size=n), 1),
            "vibration_mm_s": np.round(rng.lognormal(mean=1.5, sigma=0.5, size=n), 2),
            "oil_pressure_kpa": np.round(rng.normal(350, 50, size=n), 1),
            "rpm": rng.integers(500, 3000, size=n),
            "operating_hours": rng.integers(100, 50_000, size=n),
            "days_since_maintenance": rng.integers(0, 365, size=n),
            "load_percentage": np.round(rng.beta(5, 2, size=n) * 100, 1),
            "ambient_temperature_celsius": np.round(rng.normal(25, 10, size=n), 1),
            "hydraulic_pressure_kpa": np.round(rng.normal(200, 30, size=n), 1),
            "num_previous_failures": rng.poisson(lam=1.5, size=n),
            "equipment_type": rng.choice(SA_MINING_EQUIPMENT, size=n),
            "mine_type": rng.choice(mine_types, size=n),
            "shift": rng.choice(shift_types, size=n),
            "province": rng.choice(SA_PROVINCES, size=n),
        })

        return df

    def _generate_target(self, df):
        """Generate failure labels (~10% failure rate) based on equipment stress signals."""
        rng = self.rng

        risk_score = (
            0.01 * df["temperature_celsius"]
            + 0.1 * df["vibration_mm_s"]
            - 0.005 * df["oil_pressure_kpa"]
            + 0.0005 * df["rpm"]
            + 0.00005 * df["operating_hours"]
            + 0.005 * df["days_since_maintenance"]
            + 0.01 * df["load_percentage"]
            + 0.1 * df["num_previous_failures"]
        )

        threshold = np.percentile(risk_score, 90)
        prob = 1 / (1 + np.exp(-5.0 * (risk_score - threshold)))
        noise = rng.normal(0, 0.01, size=len(df))
        prob = np.clip(prob + noise, 0.001, 0.85)

        return (rng.random(len(df)) < prob).astype(int)


if __name__ == "__main__":
    output_path = MINING_DIR / "data" / "equipment_data.csv"
    generator = EquipmentFailureDataGenerator(n_samples=15_000)
    df = generator.generate_and_save(output_path)
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
