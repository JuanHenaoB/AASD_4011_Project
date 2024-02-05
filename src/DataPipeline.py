from src.Interfaces import FeatureExtractionStep, PreprocessingStep


class DataPipeline:
    """
    The datapipe line class is used to configure a pipline
    This is part of a chosen OOP that uses DI and the straigty pattern


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

    def __init__(self, preprocessors=None, feature_extractors=None):
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.feature_extractors = (
            feature_extractors if feature_extractors is not None else []
        )

    def add_preprocessor(self, preprocessor: PreprocessingStep):
        self.preprocessors.append(preprocessor)

    def add_feature_extractor(self, extractor: FeatureExtractionStep):
        self.feature_extractors.append(extractor)

    def execute(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.apply(data)
        for extractor in self.feature_extractors:
            data = extractor.extract(data)
        return data
