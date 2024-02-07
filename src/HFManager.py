from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

class HFManager:
    """A class that manages the training and evaluation of Hugging Face models.

    Attributes:
        training_args (TrainingArguments): The training arguments for the models.
        models (dict): A dictionary that stores the Hugging Face models.
        ckpt (str): The checkpoint name for the models.
        trainers (dict): A dictionary that stores the Hugging Face trainers.
    """

    def __init__(self):
        self.training_args = None
        self.models = {}
        self.ckpt = "-base-uncased"
        self.trainers = {}
    
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
        self.training_args = TrainingArguments(output_dir='training_dir',
                                  evaluation_strategy='epoch',
                                  save_strategy='epoch',
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=64,
                                  )
    
    def add_model(self, name):
        """Add a Hugging Face model to the manager.

        Args:
            name (str): The name of the model.
        """
        self.models[name] = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels = 3)

    def get_trainable_layers(self,name):
        """Print the trainable layers of a model.

        Args:
            name (str): The name of the model.
        """
        for name, param in self.models[name].named_parameters():
            print(name, param.requires_grad)
    
    def add_trainer(self, name, hf):
        """Add a Hugging Face trainer to the manager.

        Args:
            name (str): The name of the trainer.
            hf (HuggingFace): The Hugging Face object.
        """
        self.trainers[name] = Trainer(self.models[name],
                  self.training_args,
                  train_dataset = hf.tokenized_text_data["train"],
                  eval_dataset = hf.tokenized_text_data["valid"],
                  tokenizer=hf.tokenizer,
                  compute_metrics=self.score_metrics)
        
    def train_trainer(self, name):
        """Train a Hugging Face trainer.

        Args:
            name (str): The name of the trainer.
        """
        import time
        start = time.time()
        self.trainers[name].train()
        end = time.time()
        print("Training execution time ",end-start, 2)