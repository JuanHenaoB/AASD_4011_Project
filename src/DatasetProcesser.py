import pandas as pd
import logging
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import DatasetDict

import tensorflow as tf


class DatasetProcessor:
    """
    Class that encapsulates methods and attributes related to preprocessing
    datasets and feature extraction

    Attributes
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    label_column : str, optional
        The name of the label column in the DataFrame.

    Methods
    -------
    one_hot_encode()
        Applies one-hot encoding to the specified DataFrame.
    balance(method='downsample')
        Balances the dataset based on the one-hot encoded label columns.
    head(n=5)
        Returns the first n rows.

    Examples
    --------
    Examples of how to use this class.
    """

    def __init__(self, df, label_column="label"):
        self.df = df.copy()
        self.label_column = label_column
        self.embeddings = []

    def check_balance(self):
        """
        Visualizes the balance of classes in the dataset.
        """
        if self.label_column and self.label_column in self.df.columns:
            self.df[self.label_column].value_counts(ascending=True).plot.barh()
            plt.title("Balance of the target classes in the dataset")
            plt.show()
            if self.label_column and self.label_column in self.df.columns:
                logging.info(self.df[self.label_column].value_counts())
            else:
                logging.warning("Label column not set or not found in DataFrame.")

    def head(self, n=5):
        """
        Returns the first n rows.
        """
        return self.df.head(n)

    def __getattr__(self, attr):
        """
        Delegate attribute access to the DataFrame object if it's not found on the Dataset instance.
        This allows direct access to DataFrame methods and properties not explicitly defined in Dataset.
        """
        return getattr(self.df, attr)

    def downsample_balance(self):
        """
        Balances the dataset by downsampling classes to the size of the smallest class,
        specifically for a column with integer labels.

        Args:
            param1 (type): Description of param1
            param2 (type): Description of param2

        Returns:
            type: Description of return value


        """

        if self.label_column and self.label_column in self.df.columns:
            # Separate data by class
            classes = self.df[self.label_column].unique()
            separated_classes = {
                label: self.df.loc[self.df[self.label_column] == label]
                for label in classes
            }

            # Find the smallest class size
            min_size = min([len(df) for df in separated_classes.values()])

            # Downsample other classes to match the smallest class size
            downsampled_classes = [
                df.iloc[0:min_size] for df in separated_classes.values()
            ]

            # Concatenate the downsampled classes into one DataFrame
            self.df = pd.concat(downsampled_classes)

            # Shuffle the rows and reset the index
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        return self

    def save_dataset(self, file_name, pre="../data/"):
        """
        Saves file_name current DataFrame to a CSV file.

        Parameters:
        - file_path (str): The path (including file name) where to save the CSV.
        """

        self.df.to_csv(pre + file_name, index=False)
        logging.info(f"DataFrame saved to {pre+file_name}")

    def save_to_csv(self, file_path):
        self.df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
        return self

    def get_list_of_news(self):
        return [self.df.loc[i, "news"].split() for i in range(len(self.df))]

    def get_list_of_list(self, list_of_lists):
        regex = re.compile("[^a-zA-Z]")

        re_list_of_lists_1 = []
        for i in range(len(list_of_lists)):
            temp = [
                regex.sub("", list_of_lists[i][token])
                for token in range(len(list_of_lists[i]))
            ]
            temp = list(filter(lambda a: a != "", temp))
            re_list_of_lists_1.append(temp)
        return re_list_of_lists_1

    def embbeding_word_feats(self, list_of_lists, word2vec_model):
        """
        Generates word embeddings for each token in the documents.

        Args:
        - list_of_lists (list of list of str): The corpus with each document as a list of tokens.
        - word2vec_model (gensim.models.KeyedVectors): Pre-trained Word2Vec model.

        Returns:
        - List of lists containing embeddings for each token in each document.
        """
        # Extract model vocabulary for faster checking
        model_vocab = set(word2vec_model.key_to_index.keys())

        # Generate embeddings using list comprehension
        feats = [
            [word2vec_model[token] for token in doc if token in model_vocab]
            for doc in list_of_lists
        ]

        return feats

    # Pad the number of tokens in documents
    def padding_embeddings(self, list_of_lists, pad_to=45):
        DIMENSION = 300  # Dimension of the vectors that represent word embeddings
        zero_vector = np.zeros(DIMENSION)  # Zero vector for padding

        # Use list comprehension for efficiency
        padded_feats = [
            (doc + [zero_vector] * (pad_to - len(doc)))[:pad_to]
            for doc in list_of_lists
        ]
        return padded_feats

    def get_embeddings(self, word2vec_model, pad_to=45):
        # Step 1: Preprocess text data to get list of lists
        list_news = self.get_list_of_news()  # Debug: print(len(list_news))
        list_of_lists = self.get_list_of_list(
            list_news
        )  # Debug: print(len(list_of_lists[0]))

        # Step 2: Generate embeddings
        raw_embeddings = self.embbeding_word_feats(
            list_of_lists, word2vec_model
        )  # Ensure method name is correct

        # Step 3: Pad embeddings
        return self.padding_embeddings(raw_embeddings, pad_to=pad_to)

    def train_val_test_split(
        self, embeddings, test_size=0.2, val_size=0.5, random_state=18
    ):
        """
        Splits the dataset into training, validation, and test sets.

        Parameters:
        - test_size: Fraction of the dataset to include in the test split (default is 0.2).
        - val_size: Fraction of the test set to include in the validation split (default is 0.5).
        - random_state: Random state for reproducibility (default is 18).

        Returns:
        - A dictionary containing the split datasets: {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
        """

        X = np.array(embeddings)
        y = tf.keras.utils.to_categorical(self.df["label"], num_classes=3)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Further split test set into test and validation sets
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=val_size, random_state=random_state
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }


    def __getattr__(self, attr):
        # Improved attribute handling
        if attr in self.__dict__:
            return getattr(self, attr)
        elif hasattr(self.df, attr):
            return getattr(self.df, attr)
        else:
            raise AttributeError(
                f"'Dataset' object and its 'df' have no attribute '{attr}'"
            )




class HFprocesser:
    def __init__(self,dataset) -> None:
        self.splits = None
        self.dataset = dataset
        self.tokenized_text_data = 'None'
        self.tokenizer = None
        print('init')
    def create_train_test_val_splits(self, test_size=0.2, val_size=0.4, seed=18):
        """
        Splits a dataset into training, testing, and validation sets.

        Args:
        - self.dataset (Dataset): The raw dataset to split.
        - test_size (float): The proportion of the dataset to include in the test split.
        - val_size (float): The proportion of the test split to include in the validation set.
        - seed (int): The random seed for reproducible splits.

        Returns:
        - DatasetDict: A dictionary containing the splits 'train', 'test', and 'valid'.
        """
        # Split raw dataset into train and test
        train_test_split = self.dataset['train'].train_test_split(test_size=test_size, seed=seed)
        
        # Further split test set into test and validation sets
        test_val_split = train_test_split['test'].train_test_split(test_size=val_size, seed=seed)
        
        # Assemble the final splits into a single DatasetDict
        self.splits = DatasetDict({
            'train': train_test_split['train'],
            'test': test_val_split['train'],  # Remaining part of the test split after carving out validation set
            'valid': test_val_split['test']   # Validation set
        })
        
        return self.splits
    
    def tokenize_text_data(self, text_column='news', tokenizer_checkpoint='distilbert-base-uncased', max_length=512):
        """
        Tokenizes the text data in the DatasetDict using a specified tokenizer.

        Parameters:
        - dataset_dict: DatasetDict, the dataset splits to tokenize.
        - text_column: str, the name of the column containing text to tokenize.
        - tokenizer_checkpoint: str, the model checkpoint for the tokenizer.
        - max_length: int, the maximum sequence length.

        Returns:
        - DatasetDict: A DatasetDict containing the tokenized splits.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.tokenizer = tokenizer
        def tokenize_function(examples):
            # Ensure that the text_column is correctly processed
            return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=max_length)
        
        # Apply tokenization across all splits in the DatasetDict
        tokenized_dataset_dict = self.splits.map(tokenize_function, batched=True)
        print(tokenized_dataset_dict)
        
        self.tokenized_text_data = tokenized_dataset_dict 
        return tokenized_dataset_dict
    

    
    