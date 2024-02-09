from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras import optimizers
import time
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



class ModelManager:
    """
    A class that manages the creation, training, and evaluation of deep learning models.

    Attributes
    ----------
    models : dict
        A dictionary that stores the trained models, their configurations, and training history.

    Methods
    -------
    run_model_config(tvt_dict, model_config)
        Run the model configuration and train an LSTM model based on the provided configuration parameters.
    plot_model_history(model_name, metrics=['loss'])
        Plot the training history for the specified metrics of a given model.
    predict(model_name, X_test, return_class_indices=True)
        Make predictions with the specified model on the provided test data.
    plot_confusion_matrix(model_name, y_true, y_pred, title='Confusion Matrix')
        Plot the confusion matrix for the given true labels and predictions.
    print_performance_metrics(model_name, y_true, y_pred, dataset_type='Data Set')
        Print accuracy and F1 score for the given true labels and predictions.
    """
  
    def __init__(self):
        self.models = {}

    def run_model_config(self,tvt_dict, model_config): 
        """
        Run the model configuration.

        Args
        ----------
        tvt_dict : dict
            A dictionary containing the training and validation data splits.
    model_config : dict
            A dictionary containing the configuration parameters for the model.

        Returns
        -------
        None

        Notes
        -----
        This method initializes and trains an LSTM model based on the provided configuration parameters.
        The trained model, configuration, and training history are stored in the `models` attribute of the class.

        Examples
        --------
        >>> tvt_dict = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}
        >>> model_config = {'model_name': 'LSTM_Model', 'units': 64, 'dropout_rate': 0.3, 'epochs': 50}
        >>> model_manager.run_model_config(tvt_dict, model_config)
        """

        
        model_name = model_config['model_name']
        input_shape=(tvt_dict['X_train'].shape[1], tvt_dict['X_train'].shape[2]) #add first layer for lstm
        units = model_config.get('units', 32)
        dropout_rate = model_config.get('dropout_rate', 0.2)
        return_sequences_config = model_config.get('return_sequences_config', False)  # Assuming single config, not a list
        dense_units = model_config.get('dense_units', 3)
        activation = model_config.get('activation', 'tanh')
        output_activation = model_config.get('output_activation', 'softmax')
        
        # Additional training configuration
        epochs = model_config.get('epochs', 30)
        batch_size = model_config.get('batch_size', 32)
        optimizer_config = model_config.get('optimizer', 'adam')
        loss = model_config.get('loss', 'categorical_crossentropy')
        metrics = model_config.get('metrics', ['accuracy'])
        
        # Initialize the LSTM model
        lstm_model = Sequential()
        
        # Add LSTM layer
        lstm_model.add(LSTM(units=units, 
                            return_sequences=return_sequences_config, 
                            input_shape=input_shape, 
                            activation=activation))
        
        # Add Dropout
        lstm_model.add(Dropout(dropout_rate))
        
        # Optional additional LSTM layers could be added here based on more sophisticated `model_config` setups
        
        # Add Dense layer for output
        lstm_model.add(Dense(dense_units, activation=output_activation))
        
        # Compile the model with optimizer configuration
        if optimizer_config == 'sgd':
            learning_rate = model_config.get('learning_rate', 0.01)
            momentum = model_config.get('momentum', 0.0)
            nesterov = model_config.get('nesterov', False)
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        else:
            optimizer = optimizer_config  # Assuming 'adam' or other Keras supported strings
        
        lstm_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Train the model
        start = time.time()
        lstm_history = lstm_model.fit(tvt_dict['X_train'], tvt_dict['y_train'], 
                                    epochs=epochs, 
                                    batch_size=batch_size, 
                                    validation_data=(tvt_dict['X_val'], tvt_dict['y_val']))
        end = time.time()
        
        # Print the training process execution time
        print(f"{model_name} Training process execution time:", round(end - start, 2))
        
        self.models[model_name] = {
            'model': lstm_model,
            'config': model_config,
            'history': lstm_history
        }



    def plot_model_history(self, model_name, metrics=['loss']):
            """
            Plot the training history for the specified metrics of a given model.

            Args:
                model_name: The name of the model to plot history for.
                metrics: A list of strings, the metrics to plot (e.g., ['loss', 'val_loss']).
            """
            if model_name not in self.models:
                print(f"Model {model_name} not found.")
                return

            history = self.models[model_name]['history']
            plt.figure()

            for metric in metrics:
                if metric in history.history:
                    values = history.history[metric]
                    epochs = range(1, len(values) + 1)
                    plt.plot(epochs, values,'bo', label=f'Training {metric}')
                else:
                    print(f"Metric {metric} not found in model history.")
                    
            # Attempt to plot validation metrics if they exist and are specified
            for metric in metrics:
                val_metric = f'val_{metric}' if 'val_' not in metric else metric
                if val_metric in history.history:
                    val_values = history.history[val_metric]
                    plt.plot(epochs, val_values, 'b',label=f'Validation {metric}')
                else:
                    print(f"Metric {val_metric} not found in model history.")

            plt.title(f'Training and Validation Metrics for {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.show()


    def predict(self, model_name, X_test, return_class_indices=True):
            """
            Make predictions with the specified model on the provided test data. 

            Args:
                model_name: The name of the model to use for predictions.
                X_test: The test data to predict on.
                return_class_indices: If True, convert softmax outputs to class indices.

            Returns:
                The predictions made by the model. If return_class_indices is True,
            returns class indices; otherwise, returns the raw predictions.
            """
            if model_name not in self.models:
                print(f"Model {model_name} not found.")
                return None

            model = self.models[model_name]['model']
            y_pred = model.predict(X_test)

            if return_class_indices:
                y_pred = np.argmax(y_pred, axis=1)
            
            return y_pred
    
    def plot_confusion_matrix(self, model_name, y_true, y_pred, title='Confusion Matrix'):
            """
            Plot the confusion matrix for the given true labels and predictions.

            Args:
                model_name: The name of the model, used for title customization.
                y_true: The true labels.
                y_pred: The predictions made by the model.
                title: The title for the plot.
            """
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f'{title} for {model_name}')
            plt.show()


    def print_performance_metrics(self, model_name, y_true, y_pred, dataset_type='Data Set'):
            """
            Print accuracy and F1 score for the given true labels and predictions.

            Args:
                model_name: The name of the model, used for title customization.
                y_true: The true labels.
                y_pred: The predictions made by the model.
                dataset_type: A string to describe the dataset (e.g., 'Imbalanced', 'Balanced').
            """
            # Convert y_true from one-hot encoded vectors to class indices if necessary
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                y_true = y_true.argmax(axis=1)

            accuracy = np.round(accuracy_score(y_true, y_pred), 2)
            f1 = np.round(f1_score(y_true, y_pred, average='weighted'), 2)  # Use 'weighted' to handle imbalanced datasets

            print(f"Accuracy {dataset_type} data set for {model_name}: {accuracy}")
            print(f"F1 Score {dataset_type} data set for {model_name}: {f1}")