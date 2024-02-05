from src.Interfaces import PreprocessingStep
import pandas as pd

"""
    This has the preprocessing as part of strategies because i decided
    to take using proper OOP techniques a bit too far and began using Design
    Patterns. 

    This class is a collection of basically actions that would commonly be applied 
    during the preprocessing step. If you want another way to do tokenization
    put it here.

    Attributes
    ----------
    attribute_name : type
        Description of the attribute.

    Methods
    -------
    method_name
        Description of the method.

    Examples
    --------
    Examples of how to use this class.
"""




class Tokenization(PreprocessingStep):
    def apply(self, data):
        # Simple example of tokenization
        return data

class Balancing(PreprocessingStep):
    def apply(self, data):
        # Assume 'data' is a DataFrame with a 'label' column
        label_counts = data['label'].value_counts()
        min_count = label_counts.min()
        
        # Separate data by label
        balanced_data = pd.concat([
            data[data.label == label].sample(n=min_count, random_state=42) for label in label_counts.index
        ])
        
        # Shuffle rows and reset index
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_data
    

class OneHotEncoding:
    def __init__(self, label_column):
        self.label_column = label_column

    def apply(self, data):
        """
        Transforms the label column in the data to a one-hot encoded format.

        Parameters:
        - data (DataFrame): The DataFrame containing the data with a categorical label column.

        Returns:
        - DataFrame: A DataFrame with the label column one-hot encoded.
        """
        # Perform one-hot encoding on the specified label column
        encoded_labels = pd.get_dummies(data[self.label_column], prefix=self.label_column)
        # Drop the original label column from the data
        data = data.drop(self.label_column, axis=1)
        # Concatenate the original data with the one-hot encoded labels
        data = pd.concat([data, encoded_labels], axis=1)

        return data
    


class BalancingOneHot:
    def apply(self, data):
        """
        Balances the dataset based on one-hot encoded labels.
        
        Parameters:
        - data (DataFrame): The DataFrame containing the features and one-hot encoded labels.
        - label_columns (list of str): The column names of the one-hot encoded labels.
        
        Returns:
        - DataFrame: A balanced DataFrame based on the one-hot encoded labels.
        """
        label_columns = [col for col in data.columns if col.startswith('label_')]
        # Calculate the sum of each one-hot encoded label column to find the count of each class
        label_counts = data[label_columns].sum()
        min_count = label_counts.min()
        
        # Initialize an empty DataFrame to store the balanced dataset
        balanced_data = pd.DataFrame(columns=data.columns)
        
        # Iterate over each label column and sample `min_count` rows for each class
        for label in label_columns:
            # Find rows where the current label column is 1 (indicating class membership)
            class_data = data[data[label] == 1]
            
            # If the class dataset is larger than `min_count`, downsample it
            if len(class_data) > min_count:
                class_data = class_data.sample(n=int(min_count), random_state=42)
            
            # Append the downsampled class dataset to the balanced dataset
            balanced_data = pd.concat([balanced_data, class_data])
        
        # Shuffle rows and reset index
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        for label in label_columns:
            balanced_data[label] = balanced_data[label].astype(int)

        return balanced_data