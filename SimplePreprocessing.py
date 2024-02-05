import pandas as pd

class SimplePreprocessing:
    def __init__(self, label_column):
        """
        A simple preprocessing class for handling labels and performing one-hot encoding.

        Parameters:
        - label_column (str): The name of the column containing the labels to be processed.
        """
        self.label_column = label_column

    def apply(self, df):
        """
        Applies preprocessing to the specified DataFrame.

        Steps:
        1. Converts -1 labels to "negative".
        2. Performs one-hot encoding on the label column.
        3. Drops the original label column and returns the modified DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame to preprocess.

        Returns:
        - DataFrame: The preprocessed DataFrame with one-hot encoded labels.
        """
        # Step 1: Convert -1 labels to "negative"
        df[self.label_column] = df[self.label_column].replace(-1, 'negative')

        # Step 2: Perform one-hot encoding
        encoded_labels = pd.get_dummies(df[self.label_column], prefix=self.label_column)

        # Step 3: Concatenate the original DataFrame (minus the label column) with the encoded labels
        df = df.drop(columns=[self.label_column])
        df = pd.concat([df, encoded_labels], axis=1)

        return df


def balance_data(self, df, label_columns):
    """
    Balances the dataset based on the one-hot encoded label columns.

    Parameters:
    - df (DataFrame): The DataFrame containing one-hot encoded labels.
    - label_columns (Index): Column names of the one-hot encoded labels.

    Returns:
    - DataFrame: A balanced DataFrame.
    """
    # Calculate the sum of each one-hot encoded label column to find the count of each class
    label_counts = df[label_columns].sum().min()

    # Initialize an empty DataFrame to store the balanced dataset
    balanced_df = pd.DataFrame()

    # Iterate over each label column to balance the dataset
    for label in label_columns:
        # Filter rows where the current label is 1
        label_df = df[df[label] == 1]

        # Sample or trim the rows for the current label to match the min_count
        label_df_balanced = label_df.sample(n=label_counts, random_state=42)

        # Append the balanced rows for the current label to the balanced_df
        balanced_df = pd.concat([balanced_df, label_df_balanced], axis=0)

    # Shuffle rows and reset index
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df