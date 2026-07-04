from datasets import load_from_disk

from src.evaluation.evaluator import Evaluator
from src.inference import LoRAInference


class Benchmark:

    def __init__(
        self,
        dataset_path: str,
        base_model: str,
        lora_path: str,
    ):

        self.dataset = load_from_disk(dataset_path)

        self.inference = LoRAInference(
            base_model=base_model,
            lora_path=lora_path,
        )

        self.evaluator = Evaluator()

    def evaluate(
        self,
        num_samples: int,
    ):

        test_dataset = self.dataset["test"].select(
            range(min(num_samples, len(self.dataset["test"])))
        )

        predictions = []
        references = []

        for example in test_dataset:

            prediction = self.inference.generate(
                system=example["system"],
                user=example["user"],
            )

            predictions.append(prediction)
            references.append(example["assistant"])

        return self.evaluator.evaluate(
            predictions,
            references,
        )