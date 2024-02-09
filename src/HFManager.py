from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#check evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class HFManager:
    """A class that manages the training and evaluation of Hugging Face models.

    Attributes:
        training_args (TrainingArguments): The training arguments for the models.
        models (dict): A dictionary that stores the Hugging Face models.
        ckpt (str): The checkpoint name for the models.
        trainers (dict): A dictionary that stores the Hugging Face trainers.
    """

    def __init__(self, ckpt ='distilbert/distilbert-base-uncased'):
        self.training_args = None
        self.models = {}
        self.ckpt = "distilbert/distilbert-base-uncased"
        self.trainers = {}
        self.predictions = {}
        self.training_history = {}

    def score_metrics(self, pred):
        """Calculate the accuracy and F1 score of the predictions.

        Args:
            pred (TrainerPrediction): The predictions made by the trainer.

        Returns:
            dict: A dictionary containing the accuracy and F1 score.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def set_trainer_args(self):
        """Set the training arguments for the models."""
        self.training_args = TrainingArguments(
            output_dir="training_dir",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
        )

    def add_model(self, name):
        """Add a Hugging Face model to the manager.

        Args:
            name (str): The name of the model.
        """
        self.models[name] = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased", num_labels=3
        )

    def get_trainable_layers(self, name):
        """Print the trainable layers of a model.

        Args:
            name (str): The name of the model.
        """
        for name, param in self.models[name].named_parameters():
            print(name, param.requires_grad)


    def freeze_layers(self, name):
        """Freeze all distilbert layers

        Args:
            name (str): The name of the model.
        """
        for name, param in self.models[name].named_parameters():
            param.requires_grad = False


    def add_trainer(self, trainer_name,model_name, hf):
        """Add a Hugging Face trainer to the manager.

        Args:
            name (str): The name of the trainer.
            hf (HuggingFace): The Hugging Face object.
        """
        self.trainers[trainer_name] = Trainer(
            self.models[model_name],
            self.training_args,
            train_dataset=hf.tokenized_text_data["train"],
            eval_dataset=hf.tokenized_text_data["valid"],
            tokenizer=hf.tokenizer,
            compute_metrics=self.score_metrics,
        )

    def train_trainer(self, name):
        """Train a Hugging Face trainer.

        Args:
            name (str): The name of the trainer.
        """
        import time

        start = time.time()
        train_result = self.trainers[name].train()
        end = time.time()

        # Convert TrainOutput to dictionary
        train_result_dict = train_result.__dict__

        # Calculate execution time
        execution_time = round(end - start, 2)

        # Store the training result and execution time in the training_history
        self.training_history[name] = {
            "train_result": train_result_dict,
            "execution_time": execution_time
        }

        print("Training execution time ", execution_time)


    def predict_trainer(self, name, split):
        """Make predictions using a Hugging Face trainer.

        Args:
            name (str): The name of the trainer.
            split (dict): The dataset split to make predictions on.
        """
        prediction_output = self.trainer[name].predict(split["test"])
        
        y_preds = np.argmax(self.pred.predictions, axis=1)
        self.predictions[name] = dict(
            {"pred_output": prediction_output, "y_preds": y_preds}
        )
        return prediction_output
    
    
    
    def plot_prediction(self, name, split):
        """Plot the confusion matrix for the predictions.

        Args:
            name (str): The name of the trainer.
            split (dict): The dataset split to plot the confusion matrix for.
        """
        cm = confusion_matrix(split["test"]["label"], self.predictions[name]["y_preds"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def print_eval(self, name, split):
        """Print the evaluation metrics for the predictions.

        Args:
            name (str): The name of the trainer.
            split (dict): The dataset split to evaluate the predictions on.
        """
        print("Accuracy :", round(accuracy_score(split["test"]["label"], self.predictions[name]["y_preds"]), 2))
        print("F1 Score :", round(f1_score(split["test"]["label"], self.predictions[name]["y_preds"], average=None), 2))
