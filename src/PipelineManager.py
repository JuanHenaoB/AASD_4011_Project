from src.DataPipeline import DataPipeline
import logging

class PipelineManager:
    """
    Creates a object to manage DataPipline objects
    

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

    def __init__(self):
        self.models = {}
        self.pipelines: dict[str, DataPipeline] = {}
        # TODO: Add other components like models and model runner

    def run_pipeline(self, name, data):
        """
        Executes a named pipeline on the provided data.

        Parameters
        ----------
        name : str
            The name of the pipeline to run.
        data : any
            The data to process with the pipeline.

        Returns
        -------
        The processed data, or None if the pipeline does not exist.
        """
        if name in self.pipelines:
            return self.pipelines[name].execute(data)
        else:
            logging.error(f"Pipeline '{name}' not found.")
            return None

    def add_pipeline(self, name, pipeline):
        """
        Adds a DataPipeline object to the PipelineManager.

        Parameters
        ----------
        name : str
            The name of the pipeline.
        pipeline : DataPipeline
            The DataPipeline object to be added.

        Returns
        -------
        None
        """
        self.pipelines[name] = pipeline