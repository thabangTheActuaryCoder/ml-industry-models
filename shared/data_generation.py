"""Base class for synthetic data generation across industries."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from shared.config import RANDOM_SEED


class BaseSyntheticDataGenerator(ABC):
    """Abstract base class for generating synthetic datasets.

    Subclasses must implement `_generate_features` and `_generate_target`.
    """

    def __init__(self, n_samples, random_seed=RANDOM_SEED):
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def _generate_features(self):
        """Generate feature columns. Must return a pandas DataFrame."""
        ...

    @abstractmethod
    def _generate_target(self, df):
        """Generate the target column given the feature DataFrame.

        Must return a pandas Series.
        """
        ...

    def generate(self):
        """Run the full generation pipeline and return a DataFrame."""
        df = self._generate_features()
        df["target"] = self._generate_target(df)
        return df

    def generate_and_save(self, output_path):
        """Generate data and save to CSV."""
        df = self.generate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        return df
