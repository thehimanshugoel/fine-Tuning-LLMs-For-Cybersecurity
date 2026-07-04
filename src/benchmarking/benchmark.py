import time

from datasets import load_from_disk

from src.config import MAX_NEW_TOKENS
from src.evaluation.evaluator import Evaluator
from src.inference import LoRAInference


class Benchmark:
    """
    Runs benchmarking for a trained model.
    """

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

        latencies = []
        token_counts = []

        benchmark_example = None

        print("Generating predictions...\n")

        for idx, sample in enumerate(test_dataset):

            start_time = time.perf_counter()

            prediction = self.inference.generate(
                system=sample["system"],
                user=sample["user"],
                max_new_tokens=MAX_NEW_TOKENS,
            )

            end_time = time.perf_counter()

            latency = end_time - start_time
            latencies.append(latency)

            generated_tokens = len(
                self.inference.tokenizer.encode(
                    prediction,
                    add_special_tokens=False,
                )
            )

            token_counts.append(generated_tokens)

            predictions.append(prediction)
            references.append(sample["assistant"])

            # Save first evaluated example
            if benchmark_example is None:
                benchmark_example = {
                    "prompt": sample["user"],
                    "reference": sample["assistant"],
                    "prediction": prediction,
                }

            print(f"[{idx + 1}/{len(test_dataset)}] Done")

        results = self.evaluator.evaluate(
            predictions,
            references,
        )

        average_latency = sum(latencies) / len(latencies)

        average_generated_tokens = (
            sum(token_counts) / len(token_counts)
        )

        tokens_per_second = (
            average_generated_tokens / average_latency
        )

        results["average_latency_seconds"] = average_latency
        results["average_generated_tokens"] = average_generated_tokens
        results["tokens_per_second"] = tokens_per_second

        return results, benchmark_example