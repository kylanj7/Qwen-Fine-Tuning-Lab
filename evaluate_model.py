#!/usr/bin/env python3
"""
Unified Model Evaluator
=======================
Evaluates fine-tuned models using configuration-driven dataset support.
Works with any dataset defined in configs/datasets/.

Features:
- Config-driven dataset support (loads from configs/datasets/*.yaml)
- DuckDuckGo web search for fact verification
- LLM-as-judge scoring via Ollama
- Categorical scoring (1.0 = Correct, 0.5 = Partial, 0.0 = Incorrect)
- Per-topic/domain breakdown

Usage:
    python evaluate_model.py --model qwen_chemistry:latest --dataset chemistry
    python evaluate_model.py --model qwen_quantum:latest --dataset quantum --max_samples 50
"""

import argparse
import json
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from datasets import load_from_disk, load_dataset
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "gpt-oss:120b"
DEFAULT_TEST_SAMPLES = 100
CONFIGS_DIR = Path(__file__).parent / "configs"


# =============================================================================
# Configuration Loading
# =============================================================================

def discover_dataset_configs() -> Dict[str, Path]:
    """Discover available dataset configurations."""
    config_dir = CONFIGS_DIR / "datasets"
    configs = {}

    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            name = yaml_file.stem
            configs[name] = yaml_file

    return configs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Web Search
# =============================================================================

class DuckDuckGoSearch:
    """Simple DuckDuckGo search client using the Instant Answer API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search DuckDuckGo and return results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of dicts with 'title', 'snippet', 'url' keys
        """
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            response = self.session.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            results = []

            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data.get("Abstract", ""),
                    "url": data.get("AbstractURL", "")
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })

            return results[:max_results]

        except Exception as e:
            print(f"    [!] Search error: {e}")
            return []


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response from an Ollama model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"    [!] Ollama error: {e}")
            return ""

    def check_model_available(self, model: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            # Check both exact match and base name match
            return model in models or any(m.startswith(model.split(':')[0]) for m in models)
        except Exception:
            return False


# =============================================================================
# Model Evaluator
# =============================================================================

class ModelEvaluator:
    """Unified evaluator for fine-tuned models."""

    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Your task is to score a model's answer to a question.

You will receive:
1. The original question
2. The model's answer
3. Reference information (may include web search results or ground truth)

Score the answer using these categories:
- CORRECT (1.0): Answer is accurate, complete, and aligns with reference material
- PARTIAL (0.5): Core concept is right but has minor errors, missing details, or incomplete explanation
- INCORRECT (0.0): Contains wrong facts, contradicts reference material, or is off-topic

Respond with ONLY a JSON object in this exact format:
{"score": <0.0 or 0.5 or 1.0>, "reason": "<brief explanation>"}

Do not include any other text."""

    def __init__(
        self,
        test_model: str,
        dataset_config: Dict[str, Any],
        test_data_path: Optional[str] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_samples: Optional[int] = DEFAULT_TEST_SAMPLES,
        output_dir: str = "evaluation_results",
        use_web_search: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            test_model: Ollama model name to evaluate
            dataset_config: Dataset configuration dictionary
            test_data_path: Path to test dataset (if pre-saved)
            judge_model: Ollama model for judging
            max_samples: Maximum samples to evaluate (None for all)
            output_dir: Directory to save results
            use_web_search: Whether to use web search for verification
        """
        self.test_model = test_model
        self.dataset_config = dataset_config
        self.test_data_path = test_data_path
        self.judge_model = judge_model
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.use_web_search = use_web_search

        self.ollama = OllamaClient()
        self.search = DuckDuckGoSearch()

        self.results = []
        self.domain_name = dataset_config.get('domain', dataset_config.get('name', 'General'))

        # Extract field mappings from config
        fields = dataset_config.get('fields', {})
        self.instruction_field = fields.get('instruction', 'instruction')
        self.response_field = fields.get('response', 'response')
        self.context_fields = fields.get('context_fields', [])

        # For MCQ datasets
        self.options_fields = fields.get('options', [])
        self.correct_option_field = fields.get('correct_option')

        os.makedirs(output_dir, exist_ok=True)

    def load_test_data(self) -> List[Dict]:
        """Load the test dataset."""
        print(f"Loading test data for: {self.dataset_config.get('name', 'Unknown')}")

        # Try loading from saved test split first
        if self.test_data_path:
            try:
                if Path(self.test_data_path).is_dir():
                    dataset = load_from_disk(self.test_data_path)
                else:
                    with open(self.test_data_path, "r") as f:
                        data = json.load(f)
                    if self.max_samples:
                        data = data[:self.max_samples]
                    print(f"Loaded {len(data)} test samples from file")
                    return data
            except Exception as e:
                print(f"Could not load from path: {e}")

        # Load from HuggingFace and use test split
        dataset_name = self.dataset_config['dataset_name']
        split = self.dataset_config.get('split', 'train')

        print(f"Loading {dataset_name} split={split}")
        dataset = load_dataset(dataset_name, split=split)

        # Apply train/val/test split if configured
        split_config = self.dataset_config.get('train_val_test_split')
        if split_config:
            test_ratio = split_config.get('test', 0.2)
            val_ratio = split_config.get('val', 0.2)
            seed = split_config.get('seed', 3407)

            # Split to get test portion
            temp_test_size = val_ratio + test_ratio
            split1 = dataset.train_test_split(test_size=temp_test_size, seed=seed)

            if test_ratio > 0:
                val_test_ratio = test_ratio / temp_test_size
                split2 = split1['test'].train_test_split(test_size=val_test_ratio, seed=seed)
                test_dataset = split2['test']
            else:
                test_dataset = split1['test']

            dataset = test_dataset
            print(f"Using test split: {len(dataset)} samples")

        # Convert to list of dicts
        data = []
        for item in dataset:
            entry = {
                "question": item.get(self.instruction_field, ""),
                "reference_answer": item.get(self.response_field, ""),
            }

            # Add context fields
            for field in self.context_fields:
                # Handle typos in field names (like "topic;" -> "topic")
                clean_field = field.rstrip(';')
                if clean_field in item:
                    entry[clean_field] = item[clean_field]
                elif field in item:
                    entry[field] = item[field]

            # Handle MCQ options
            if self.options_fields:
                entry['options'] = [item.get(opt, "") for opt in self.options_fields]
                if self.correct_option_field and self.correct_option_field in item:
                    entry['correct_option'] = item[self.correct_option_field]

            data.append(entry)

        if self.max_samples:
            data = data[:self.max_samples]

        print(f"Prepared {len(data)} test samples")
        return data

    def get_model_response(self, question: str) -> str:
        """Get response from the fine-tuned model."""
        return self.ollama.generate(
            model=self.test_model,
            prompt=question,
            temperature=0.1,
            max_tokens=1024
        )

    def get_web_references(self, question: str) -> str:
        """Search web for reference information."""
        if not self.use_web_search:
            return ""

        search_query = f"{self.domain_name} {question[:150]}"
        results = self.search.search(search_query, max_results=3)

        if not results:
            return "No web references found."

        references = []
        for r in results:
            references.append(f"Source: {r['title']}\n{r['snippet']}")

        return "\n\n".join(references)

    def build_judge_context(self, item: Dict, model_answer: str, web_references: str) -> str:
        """Build the reference context for the judge."""
        context_parts = []

        # Add ground truth if available
        if item.get('reference_answer'):
            context_parts.append(f"Ground Truth Answer:\n{item['reference_answer']}")

        # Add MCQ correct answer if available
        if 'options' in item and 'correct_option' in item:
            correct_idx = item['correct_option']
            if isinstance(correct_idx, int) and 0 <= correct_idx < len(item['options']):
                letter = chr(65 + correct_idx)
                context_parts.append(f"Correct Option: {letter}) {item['options'][correct_idx]}")

        # Add web references
        if web_references:
            context_parts.append(f"Web References:\n{web_references}")

        return "\n\n".join(context_parts) if context_parts else "No reference information available."

    def judge_answer(
        self,
        question: str,
        model_answer: str,
        reference_context: str
    ) -> Tuple[float, str]:
        """Use LLM judge to score the model's answer."""
        judge_prompt = f"""Question:
{question}

Model's Answer:
{model_answer}

Reference Information:
{reference_context}

Evaluate the model's answer and provide your score."""

        response = self.ollama.generate(
            model=self.judge_model,
            prompt=judge_prompt,
            system=self.JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256
        )

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                score = float(result.get("score", 0.0))
                reason = result.get("reason", "No reason provided")

                if score not in [0.0, 0.5, 1.0]:
                    score = 0.0

                return score, reason
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [!] Failed to parse judge response: {e}")

        return 0.0, "Failed to parse judge response"

    def evaluate_single(self, item: Dict, index: int) -> Dict:
        """Evaluate a single test item."""
        question = item["question"]

        # Get topic/domain info for reporting
        topic = item.get("topic", item.get("sub_topic", self.domain_name))

        print(f"\n[{index}] Domain: {self.domain_name} / Topic: {topic}")
        print(f"    Question: {question[:100]}...")

        # Get model response
        print("    Getting model response...")
        model_answer = self.get_model_response(question)

        if not model_answer:
            print("    [!] No response from model")
            return {
                "index": index,
                "question": question,
                "topic": topic,
                "model_answer": "",
                "reference_context": "",
                "score": 0.0,
                "reason": "No response from model"
            }

        print(f"    Response: {model_answer[:100]}...")

        # Get web references
        if self.use_web_search:
            print("    Searching web for verification...")
        web_references = self.get_web_references(question)

        # Build reference context
        reference_context = self.build_judge_context(item, model_answer, web_references)

        # Judge the answer
        print(f"    Judging answer with {self.judge_model}...")
        score, reason = self.judge_answer(question, model_answer, reference_context)

        score_label = {1.0: "CORRECT", 0.5: "PARTIAL", 0.0: "INCORRECT"}[score]
        print(f"    Score: {score_label} ({score}) - {reason[:80]}...")

        return {
            "index": index,
            "question": question,
            "topic": topic,
            "model_answer": model_answer,
            "reference_answer": item.get("reference_answer", ""),
            "reference_context": reference_context,
            "score": score,
            "reason": reason
        }

    def calculate_metrics(self) -> Dict:
        """Calculate evaluation metrics."""
        total = len(self.results)
        if total == 0:
            return {}

        scores = [r["score"] for r in self.results]

        correct = sum(1 for s in scores if s == 1.0)
        partial = sum(1 for s in scores if s == 0.5)
        incorrect = sum(1 for s in scores if s == 0.0)

        raw_accuracy = (sum(scores) / total) * 100

        # Calculate by topic
        topic_metrics = {}
        for result in self.results:
            topic = result["topic"]
            if topic not in topic_metrics:
                topic_metrics[topic] = {"scores": [], "count": 0}
            topic_metrics[topic]["scores"].append(result["score"])
            topic_metrics[topic]["count"] += 1

        for topic, data in topic_metrics.items():
            data["accuracy"] = (sum(data["scores"]) / len(data["scores"])) * 100

        return {
            "total_samples": total,
            "correct": correct,
            "correct_pct": (correct / total) * 100,
            "partial": partial,
            "partial_pct": (partial / total) * 100,
            "incorrect": incorrect,
            "incorrect_pct": (incorrect / total) * 100,
            "raw_accuracy": raw_accuracy,
            "by_topic": topic_metrics
        }

    def generate_report(self, metrics: Dict) -> str:
        """Generate a formatted evaluation report."""
        report = []
        report.append("=" * 60)
        report.append(f"{self.domain_name.upper()} MODEL EVALUATION RESULTS")
        report.append("=" * 60)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Model: {self.test_model}")
        report.append(f"Judge Model: {self.judge_model}")
        report.append(f"Dataset: {self.dataset_config.get('dataset_name', 'Unknown')}")
        report.append(f"Total Samples: {metrics['total_samples']}")
        report.append(f"Web Search: {'Enabled' if self.use_web_search else 'Disabled'}")
        report.append("")
        report.append("-" * 60)
        report.append("OVERALL SCORES")
        report.append("-" * 60)
        report.append(f"Fully Correct:     {metrics['correct']:>5} ({metrics['correct_pct']:.1f}%)")
        report.append(f"Partially Correct: {metrics['partial']:>5} ({metrics['partial_pct']:.1f}%)")
        report.append(f"Incorrect:         {metrics['incorrect']:>5} ({metrics['incorrect_pct']:.1f}%)")
        report.append("")
        report.append(f"RAW ACCURACY SCORE: {metrics['raw_accuracy']:.1f}%")
        report.append("")
        report.append("-" * 60)
        report.append("SCORES BY TOPIC")
        report.append("-" * 60)

        sorted_topics = sorted(
            metrics["by_topic"].items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        for topic, data in sorted_topics:
            topic_display = str(topic)[:35] if topic else "Unknown"
            report.append(f"{topic_display:<35} {data['accuracy']:>6.1f}% (n={data['count']})")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def save_results(self, metrics: Dict, report: str):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_slug = self.domain_name.lower().replace(" ", "_")

        results_file = os.path.join(
            self.output_dir,
            f"eval_{domain_slug}_{timestamp}.json"
        )
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "test_model": self.test_model,
                    "judge_model": self.judge_model,
                    "dataset": self.dataset_config.get('dataset_name'),
                    "domain": self.domain_name,
                    "max_samples": self.max_samples,
                    "web_search": self.use_web_search
                },
                "metrics": metrics,
                "detailed_results": self.results
            }, f, indent=2)
        print(f"Detailed results saved to: {results_file}")

        report_file = os.path.join(
            self.output_dir,
            f"eval_{domain_slug}_{timestamp}.txt"
        )
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_file}")

    def run(self):
        """Run the full evaluation pipeline."""
        print("=" * 60)
        print(f"MODEL EVALUATOR - {self.domain_name.upper()}")
        print("=" * 60)

        # Check models are available
        print(f"\nChecking model availability...")
        if not self.ollama.check_model_available(self.test_model):
            print(f"[!] Test model '{self.test_model}' not found in Ollama")
            print("    Run: ollama pull <model_name>")
            return
        print(f"  Test model '{self.test_model}': OK")

        if not self.ollama.check_model_available(self.judge_model):
            print(f"[!] Judge model '{self.judge_model}' not found in Ollama")
            print(f"    Run: ollama pull {self.judge_model}")
            return
        print(f"  Judge model '{self.judge_model}': OK")

        # Load test data
        test_data = self.load_test_data()

        if not test_data:
            print("[!] No test data loaded")
            return

        # Run evaluation
        print(f"\nStarting evaluation of {len(test_data)} samples...")
        print("=" * 60)

        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            result = self.evaluate_single(item, i + 1)
            self.results.append(result)
            time.sleep(0.5)

        # Calculate metrics
        print("\n\nCalculating metrics...")
        metrics = self.calculate_metrics()

        # Generate report
        report = self.generate_report(metrics)
        print("\n" + report)

        # Save results
        self.save_results(metrics, report)

        print("\nEvaluation complete!")
        return metrics


# =============================================================================
# CLI Interface
# =============================================================================

def list_datasets():
    """List available dataset configurations."""
    configs = discover_dataset_configs()
    print("\nAvailable dataset configurations:")
    print("-" * 40)
    for name, path in configs.items():
        config = load_config(path)
        display_name = config.get('name', name)
        dataset_name = config.get('dataset_name', '')
        print(f"  {name:<20} {display_name} ({dataset_name})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned models with configurable datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --list-datasets
  python evaluate_model.py --model qwen_chemistry:latest --dataset chemistry
  python evaluate_model.py --model my_model:latest --dataset quantum --max_samples 50
  python evaluate_model.py --model my_model:latest --dataset chemistry --test_data outputs/test_dataset
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset config name (from configs/datasets/)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="Path to pre-saved test dataset (optional)"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model to use as judge (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help=f"Maximum samples to evaluate (default: {DEFAULT_TEST_SAMPLES}, use -1 for all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)"
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search for verification (faster, uses only ground truth)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available dataset configurations and exit"
    )

    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (use --list-datasets to see options)")

    # Load dataset config
    dataset_configs = discover_dataset_configs()
    if args.dataset not in dataset_configs:
        print(f"Error: Dataset '{args.dataset}' not found")
        list_datasets()
        sys.exit(1)

    dataset_config = load_config(dataset_configs[args.dataset])

    max_samples = None if args.max_samples == -1 else args.max_samples

    evaluator = ModelEvaluator(
        test_model=args.model,
        dataset_config=dataset_config,
        test_data_path=args.test_data,
        judge_model=args.judge_model,
        max_samples=max_samples,
        output_dir=args.output_dir,
        use_web_search=not args.no_web_search,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
