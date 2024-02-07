from src.Interfaces import FeatureExtractionStep, PreprocessingStep
#

class DataPipeline:
    """
    I do not know why i did this to myself.

    
    This class allows for the flexible construction of a data processing pipeline by adding
    preprocessing steps and feature extraction steps. Each step is applied in the order they
    were added to transform or extract features from the input data.

    REF: https://refactoring.guru/design-patterns/strategy
    REF: https://en.wikipedia.org/wiki/Dependency_injection

    Attributes
    ----------
    attribute_name : type
        Description of the attribute.

    Methods
    -------
    method_name
        add_preprocessor(preprocessor: PreprocessingStep): Adds a preprocessing step to the pipeline.
        add_feature_extractor(extractor: FeatureExtractionStep): Adds a feature extraction step to the pipeline.
        execute(data): Applies all preprocessing and feature extraction steps to the input data.


    Examples
    --------
    Examples of how to use this class.
    """

    def __init__(self, preprocessors=None, feature_extractors=None):
        """
        Initializes the DataPipeline with optional lists of preprocessors and feature extractors.

        Args:
            preprocessors (list of PreprocessingStep, optional): Initial preprocessing steps to add to the pipeline.
            feature_extractors (list of FeatureExtractionStep, optional): Initial feature extraction steps to add to the pipeline.
        """
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.feature_extractors = feature_extractors if feature_extractors is not None else []

    def add_preprocessor(self, preprocessor):
        """
        Adds a preprocessing step to the pipeline.

        Args:
            preprocessor (PreprocessingStep): The preprocessing step to add.

        Returns:
            None
        """
        self.preprocessors.append(preprocessor)

    def add_feature_extractor(self, extractor):
        """
        Adds a feature extraction step to the pipeline.

        Args:
            extractor (FeatureExtractionStep): The feature extraction step to add.

        Returns:
            None
        """
        self.feature_extractors.append(extractor)

    def execute(self, data):
        """
        Applies all preprocessing and feature extraction steps to the input data in the order they were added.

        Args:
            data: The input data to process. The exact type of this parameter depends on the specific preprocessors and feature extractors used.

        Returns:
            The processed data after all preprocessing and feature extraction steps have been applied. The exact type of the return value depends on the specific feature extractors used.
        """
        for preprocessor in self.preprocessors:
            data = preprocessor.apply(data)
        for extractor in self.feature_extractors:
            data = extractor.extract(data)
        return data