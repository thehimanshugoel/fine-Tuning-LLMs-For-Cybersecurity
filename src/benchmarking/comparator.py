class ModelComparator:
    """
    Compare benchmark results from multiple models.
    """

    def __init__(self):
        self.results = {}

    def add_result(
        self,
        model_name: str,
        metrics: dict,
    ) -> None:

        self.results[model_name] = metrics

    def get_results(self):

        return self.results