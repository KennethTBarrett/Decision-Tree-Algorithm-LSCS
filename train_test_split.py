import random


def train_test_split(df, test_size):
        """Randomly selects data, and splits it based upon
        the specified test size.
        Usage: `train_test_split(df, test_size)`
        Returns training and testing data.
        If a float is input as test_size, it's treated as percentage."""

        # If our test_size is a float, we're going to treat it as a percentage.
        # We need to calculate and round, because k below requires an integer.
        if isinstance(test_size, float):
                test_size = round(test_size * len(df))

        # Sampling test indices.
        test_idx = random.sample(population=df.index.tolist(), k=test_size)

        train_df = df.drop(test_idx)  # Training data / test indices dropped.
        test_df = df.loc[test_idx]  # Testing data / only test indices.

        return train_df, test_df
