#!/usr/bin/env python3
"""
Model Evaluator
===============
Evaluate fine-tuned GGUF models using LLM-as-judge scoring.

Simply run:
    python evaluate_model.py

The script will guide you through selecting:
1. Model to evaluate (from models/gguf/)
2. Dataset to test against
3. Judge model (from Ollama)
4. Number of samples
"""

import argparse
import json
import os
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests
from datasets import load_dataset
from llama_cpp import Llama

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
CONFIGS_DIR = Path(__file__).parent / "configs"
MODELS_DIR = Path(__file__).parent / "models" / "gguf"
OUTPUT_DIR = Path(__file__).parent / "evaluation_results"

# Hard-coded evaluation settings
JUDGE_MODEL = "gpt-oss:120b"
RAG_MAX_PAPERS = 5
MAX_RETRIES_ON_ZERO = 2  # Retry questions that score zero due to errors


# =============================================================================
# Ollama Client (for judge model)
# =============================================================================

class OllamaClient:
    """Client for Ollama API (used for judge model)."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 200, max_retries: int = 3) -> Optional[str]:
        """Generate a response with retry logic."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": 60,
            }
        }

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=300
                )

                # Handle 500 errors with retry
                if response.status_code == 500:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[Server error, retrying in {wait_time}s... ({attempt + 1}/{max_retries})]", end=" ", flush=True)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json().get("response", "")

            except requests.exceptions.Timeout:
                wait_time = 2 ** (attempt + 1)
                print(f"[Timeout, retrying in {wait_time}s... ({attempt + 1}/{max_retries})]", end=" ", flush=True)
                time.sleep(wait_time)
                continue
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[Error: {e}, retrying in {wait_time}s...]", end=" ", flush=True)
                    time.sleep(wait_time)
                    continue
                print(f"[Error after {max_retries} attempts: {e}]")
                return None

        print(f"[Failed after {max_retries} attempts]")
        return None


# =============================================================================
# Semantic Scholar RAG Client
# =============================================================================

class SemanticScholarRAG:
    """RAG client using Semantic Scholar API for grounding judge in academic truth."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_FIELDS = "paperId,title,abstract,year,citationCount,authors,openAccessPdf,isOpenAccess"

    def __init__(self, max_papers: int = 3, cache_enabled: bool = True, api_key: Optional[str] = None):
        self.max_papers = max_papers
        self.cache_enabled = cache_enabled
        self.session = requests.Session()

        # Use API key from param, env var, or run unauthenticated
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.session.headers.update(headers)

        self._cache: Dict[str, List[Dict]] = {}
        self._last_request_time = 0.0
        # Authenticated: 1 req/sec, Unauthenticated: ~100 req/5min
        self._min_request_interval = 1.1 if self.api_key else 3.0

    def _extract_keywords(self, question: str, max_keywords: int = 5) -> str:
        """Extract key terms from a question for search."""
        import re

        # Remove LaTeX math expressions
        text = re.sub(r'\$[^$]+\$', ' ', question)  # inline math $...$
        text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)  # display math \[...\]
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)  # \command{...}
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # \command
        text = re.sub(r'[{}_^<>\\|]', ' ', text)  # LaTeX special chars

        # Remove common question words and stopwords
        stopwords = {
            'what', 'which', 'how', 'why', 'when', 'where', 'who', 'whom',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
            'this', 'that', 'these', 'those', 'it', 'its', 'can', 'may', 'might',
            'explain', 'describe', 'define', 'discuss', 'compare', 'contrast',
            'give', 'provide', 'list', 'name', 'identify', 'answer', 'question',
            'following', 'example', 'examples', 'please', 'briefly', 'detail',
            'given', 'assume', 'using', 'use', 'find', 'denotes', 'all', 'pairs',
            'sum', 'sigma', 'beta', 'left', 'right', 'frac', 'text',
        }

        # Clean and tokenize
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
        words = text.split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique_keywords.append(w)

        # Take most relevant keywords (prioritize longer, more specific terms)
        unique_keywords.sort(key=len, reverse=True)
        selected = unique_keywords[:max_keywords]

        return ' '.join(selected)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def search_papers(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """Search Semantic Scholar for relevant papers with retry logic."""
        if limit is None:
            limit = self.max_papers

        # Check cache
        cache_key = f"{query}:{limit}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._rate_limit()

                encoded_query = quote_plus(query)
                url = f"{self.BASE_URL}/paper/search?query={encoded_query}&limit={limit}&fields={self.SEARCH_FIELDS}"

                response = self.session.get(url, timeout=10)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8 seconds
                    print(f"[Rate limited, waiting {wait_time}s...]", end=" ", flush=True)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()
                papers = data.get("data", [])

                # Filter papers with abstracts and sort by citation count
                papers_with_abstracts = [p for p in papers if p.get("abstract")]
                papers_with_abstracts.sort(key=lambda p: p.get("citationCount", 0), reverse=True)

                result = papers_with_abstracts[:limit]

                if self.cache_enabled:
                    self._cache[cache_key] = result

                return result

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[Semantic Scholar API error: {e}]")
                return []
            except Exception as e:
                print(f"[RAG search error: {e}]")
                return []

        return []

    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant academic context for a question."""
        keywords = self._extract_keywords(question)
        if not keywords:
            return ""

        papers = self.search_papers(keywords)
        if not papers:
            return ""

        # Format context from papers
        context_parts = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown")
            year = paper.get("year", "N/A")
            abstract = paper.get("abstract", "")
            citations = paper.get("citationCount", 0)

            authors = paper.get("authors", [])
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += " et al."

            # Truncate abstract if too long
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            context_parts.append(
                f"[{i}] {title} ({year})\n"
                f"    Authors: {author_names}\n"
                f"    Citations: {citations}\n"
                f"    Abstract: {abstract}"
            )

        return "\n\n".join(context_parts)

    def get_grounded_context(self, question: str, reference: str = "") -> str:
        """Get academic context to ground the judge's evaluation."""
        # Search based on question
        question_context = self.retrieve_context(question)

        # Optionally also search based on key terms in reference
        reference_context = ""
        if reference and len(reference) > 50:
            ref_keywords = self._extract_keywords(reference[:300])
            if ref_keywords:
                ref_papers = self.search_papers(ref_keywords, limit=2)
                if ref_papers:
                    reference_context = self.retrieve_context(reference[:300])

        if question_context and reference_context:
            return f"Academic context from question:\n{question_context}\n\nAdditional context from reference topic:\n{reference_context}"
        elif question_context:
            return question_context
        elif reference_context:
            return reference_context
        else:
            return ""


# =============================================================================
# GGUF Model Wrapper
# =============================================================================

class GGUFModel:
    """Wrapper for GGUF model inference with streaming."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        print(f"\nLoading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("Model loaded!\n")

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1):
        """Generate with streaming output."""
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        try:
            stream = self.llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True,
            )

            full_response = ""
            for output in stream:
                chunk = output["choices"][0]["text"]
                full_response += chunk
                print(chunk, end="", flush=True)

            print()  # newline after streaming
            return full_response.strip()

        except Exception as e:
            print(f"[Generation error: {e}]")
            return ""


# =============================================================================
# Interactive Selection
# =============================================================================

def print_header():
    print()
    print("=" * 70)
    print("MODEL EVALUATOR")
    print("=" * 70)
    print()


def select_from_list(options: List[str], prompt: str) -> str:
    print(f"{prompt}")
    print("-" * 50)

    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    print()
    while True:
        try:
            choice = input(f"Select [1-{len(options)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"Please enter 1-{len(options)}")
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def get_number_input(prompt: str, default: int, min_val: int = 1, max_val: int = 10000) -> int:
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            if not value:
                return default
            num = int(value)
            if min_val <= num <= max_val:
                return num
            print(f"Enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    try:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        return value in ('y', 'yes')
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


# =============================================================================
# Discovery Functions
# =============================================================================

def find_gguf_models() -> List[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.gguf"))


def load_dataset_configs() -> Dict[str, Dict]:
    config_dir = CONFIGS_DIR / "datasets"
    configs = {}
    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                configs[yaml_file.stem] = config
    return configs


# =============================================================================
# Evaluator
# =============================================================================

class Evaluator:
    """Model evaluator with streaming, logging, and RAG-grounded fact-checking."""

    JUDGE_PROMPT_RAG = """You are a rigorous fact-checker evaluating a model's response using academic sources as ground truth.

QUERY:
{question}

MODEL OUTPUT:
{answer}

REFERENCE ANSWER:
{reference}

ACADEMIC SOURCES (ground truth):
{rag_context}

Score the model output on three dimensions (0-100 each):

1. FACTUAL_ACCURACY: Do the claims match the academic sources and reference?
   - 100: All claims verified and accurate
   - 70-99: Minor inaccuracies that don't affect core understanding
   - 40-69: Some significant errors but partial correctness
   - 0-39: Major factual errors or contradictions

2. COMPLETENESS: Does the response fully address the query?
   - 100: Comprehensive, addresses all aspects
   - 70-99: Addresses main points, minor omissions
   - 40-69: Partial coverage, missing key elements
   - 0-39: Severely incomplete or off-topic

3. TECHNICAL_PRECISION: Are equations, terminology, and methods correct?
   - 100: Flawless technical presentation
   - 70-99: Minor notation or terminology issues
   - 40-69: Some technical errors but approach is sound
   - 0-39: Fundamental technical mistakes

Respond in this exact format:
FACTUAL_ACCURACY: [score]
COMPLETENESS: [score]
TECHNICAL_PRECISION: [score]
JUSTIFICATION: [brief explanation citing specific issues or strengths]"""

    def __init__(
        self,
        model: GGUFModel,
        model_name: str,
        judge_client: OllamaClient,
        dataset_config: Dict,
        max_samples: int,
        judge_model: str = None,
        output_json: bool = False,
        wandb_run=None,
    ):
        self.model = model
        self.model_name = model_name
        self.judge = judge_client
        self.judge_model = judge_model or JUDGE_MODEL
        self.dataset_config = dataset_config
        self.max_samples = max_samples
        self.output_json = output_json
        self.wandb_run = wandb_run

        # Initialize RAG client (always enabled)
        self.rag = SemanticScholarRAG(max_papers=RAG_MAX_PAPERS)

        self.results = []
        self.domain = dataset_config.get('domain', dataset_config.get('name', 'General'))

        fields = dataset_config.get('fields', {})
        self.instruction_field = fields.get('instruction', 'instruction')
        self.response_field = fields.get('response', 'response')

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = OUTPUT_DIR / f"eval_{self.domain.lower()}_{timestamp}.txt"
        self.json_path = OUTPUT_DIR / f"eval_{self.domain.lower()}_{timestamp}.json"
        self.articles_log_path = OUTPUT_DIR / f"articles_{self.domain.lower()}_{timestamp}.json"

        # Store article logs for manual verification
        self.article_logs = []

    def log(self, text: str, also_print: bool = False):
        """Write to log file."""
        with open(self.log_path, 'a') as f:
            f.write(text + "\n")
        if also_print:
            print(text)

    def load_test_data(self) -> List[Dict]:
        dataset_name = self.dataset_config['dataset_name']
        split = self.dataset_config.get('split', 'train')

        print(f"Loading {dataset_name}...")
        self.log(f"Loading dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split=split)

        # Apply split if configured
        split_config = self.dataset_config.get('train_val_test_split')
        if split_config:
            test_ratio = split_config.get('test', 0.2)
            val_ratio = split_config.get('val', 0.2)
            seed = split_config.get('seed', 3407)

            temp_size = val_ratio + test_ratio
            split1 = dataset.train_test_split(test_size=temp_size, seed=seed)

            if test_ratio > 0:
                ratio = test_ratio / temp_size
                split2 = split1['test'].train_test_split(test_size=ratio, seed=seed)
                dataset = split2['test']
            else:
                dataset = split1['test']

        data = []
        for item in dataset:
            data.append({
                "question": item.get(self.instruction_field, ""),
                "reference": item.get(self.response_field, ""),
                "topic": item.get("topic", item.get("sub_topic", self.domain)),
            })

        data = data[:self.max_samples]
        print(f"Loaded {len(data)} test samples\n")
        self.log(f"Test samples: {len(data)}\n")
        return data

    def evaluate_one(self, item: Dict, idx: int, total: int, output_json: bool = False) -> Dict:
        question = item["question"]
        reference = item.get("reference", "")

        # Progress marker (for backend parsing)
        print(f"Evaluating sample {idx}/{total}", flush=True)

        # Header
        header = f"\n{'='*70}\n[{idx}/{total}] QUESTION\n{'='*70}"
        print(header)
        self.log(header)

        # Show full question
        print(question)
        self.log(question)

        # Model response header
        response_header = f"\n{'-'*70}\nMODEL RESPONSE\n{'-'*70}"
        print(response_header)
        self.log(response_header)

        # Stream model response (use higher token limit for complex technical questions)
        answer = self.model.generate_stream(question, max_tokens=3072, temperature=0.1)
        self.log(answer)

        if not answer:
            self.log("\n[No response from model]")
            return {"index": idx, "score": 0.0, "reason": "No response", **item, "answer": ""}

        # RAG retrieval for fact-checking (always enabled)
        rag_header = f"\n{'-'*70}\nRAG RETRIEVAL (Semantic Scholar)\n{'-'*70}"
        print(rag_header)
        self.log(rag_header)

        print("Retrieving academic sources...", end=" ", flush=True)

        # Get papers for article logging (before get_grounded_context which just returns formatted text)
        keywords = self.rag._extract_keywords(question)
        papers = self.rag.search_papers(keywords) if keywords else []

        rag_context = self.rag.get_grounded_context(question, reference)

        # Log articles for manual verification
        article_entry = {
            "question_index": idx,
            "question": question[:500],
            "search_keywords": keywords,
            "papers_retrieved": []
        }

        for paper in papers:
            # Extract open access PDF URL if available
            open_access_pdf = paper.get("openAccessPdf")
            pdf_url = open_access_pdf.get("url") if open_access_pdf else None

            article_entry["papers_retrieved"].append({
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors", [])[:5]],
                "citation_count": paper.get("citationCount", 0),
                "abstract": paper.get("abstract", "")[:1000],
                "semantic_scholar_url": f"https://www.semanticscholar.org/paper/{paper.get('paperId')}",
                "is_open_access": paper.get("isOpenAccess", False),
                "pdf_url": pdf_url,
            })

        self.article_logs.append(article_entry)

        if rag_context:
            print(f"found {len(papers)} sources")
            self.log(f"Retrieved context:\n{rag_context}")
        else:
            print("no sources found - SKIPPING QUESTION")
            self.log("No RAG context retrieved - skipping question to avoid ungrounded scoring")
            skip_msg = f"\n>>> SKIPPED: No relevant academic sources found for grounded evaluation\n"
            print(skip_msg)
            self.log(skip_msg)
            skipped_result = {
                "index": idx,
                "question": question,
                "answer": answer,
                "reference": reference,
                "topic": item.get("topic", ""),
                "skipped": True,
                "reason": "No Semantic Scholar articles found",
            }
            self.results.append(skipped_result)
            return skipped_result

        # Judge fact-check
        judge_header = f"\n{'-'*70}\nJUDGE FACT-CHECK ({self.judge_model})\n{'-'*70}"
        print(judge_header)
        self.log(judge_header)

        judge_prompt = self.JUDGE_PROMPT_RAG.format(
            question=question[:1500],
            answer=answer[:2000],
            reference=reference[:1500] if reference else "No reference provided",
            rag_context=rag_context[:3000] if rag_context else "No academic sources retrieved"
        )

        print("Fact-checking...", end=" ", flush=True)
        judge_response = self.judge.generate(self.judge_model, judge_prompt, max_tokens=4096)

        # Handle judge failure
        if judge_response is None:
            print("\n[Judge failed - skipping question]")
            self.log("[Judge failed after retries - skipping question]")
            skip_msg = f"\n>>> SKIPPED: Judge model failed to respond\n"
            print(skip_msg)
            self.log(skip_msg)
            skipped_result = {
                "index": idx,
                "question": question,
                "answer": answer,
                "reference": reference,
                "topic": item.get("topic", ""),
                "skipped": True,
                "reason": "Judge model failed",
            }
            self.results.append(skipped_result)
            return skipped_result

        # Show raw judge response
        print(f"\n{judge_response}")

        # Log judge response (already printed via streaming)
        self.log(f"[Judge response: {judge_response}]")

        # Parse multi-dimensional scores
        scores = {"factual_accuracy": 0, "completeness": 0, "technical_precision": 0}
        justification = "Parse error"

        import re
        # Extract scores using regex
        for key, pattern in [
            ("factual_accuracy", r"FACTUAL_ACCURACY:\s*(\d+)"),
            ("completeness", r"COMPLETENESS:\s*(\d+)"),
            ("technical_precision", r"TECHNICAL_PRECISION:\s*(\d+)"),
        ]:
            match = re.search(pattern, judge_response, re.IGNORECASE)
            if match:
                scores[key] = min(100, max(0, int(match.group(1))))

        # Extract justification (preserve full text for logging)
        just_match = re.search(r"JUSTIFICATION:\s*(.+)", judge_response, re.IGNORECASE | re.DOTALL)
        if just_match:
            justification = just_match.group(1).strip()

        # Calculate weighted average (50% accuracy, 30% completeness, 20% precision)
        overall_score = (
            scores["factual_accuracy"] * 0.50 +
            scores["completeness"] * 0.30 +
            scores["technical_precision"] * 0.20
        )

        # Display scores
        score_display = f"""
>>> SCORES:
    Factual Accuracy:    {scores['factual_accuracy']:3d}/100
    Completeness:        {scores['completeness']:3d}/100
    Technical Precision: {scores['technical_precision']:3d}/100
    ─────────────────────────────
    OVERALL:             {overall_score:5.1f}/100

>>> JUSTIFICATION:
    {justification}"""
        print(score_display)
        self.log(score_display)

        # Machine-readable score line (for backend parsing)
        print(f"SCORES: factual_accuracy={scores['factual_accuracy']}, completeness={scores['completeness']}, technical_precision={scores['technical_precision']}", flush=True)

        # Build RAG sources from article logs for this question
        rag_sources = []
        if self.article_logs:
            last_article = self.article_logs[-1]
            if last_article.get("question_index") == idx:
                for paper in last_article.get("papers_retrieved", []):
                    rag_sources.append({
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "authors": paper.get("authors", []),
                        "url": paper.get("semantic_scholar_url", ""),
                        "pdf_url": paper.get("pdf_url"),
                        "is_open_access": paper.get("is_open_access", False),
                    })

        # Running tally
        result_entry = {
            "index": idx,
            "question": question,
            "answer": answer,
            "reference": reference,
            "scores": scores,
            "overall_score": overall_score,
            "justification": justification,
            "topic": item.get("topic", ""),
            "skipped": False,
            "rag_sources": rag_sources,
        }
        self.results.append(result_entry)

        # Calculate running average (exclude skipped questions)
        scored_results = [r for r in self.results if not r.get("skipped", False)]
        running_scores = [r["overall_score"] for r in scored_results]
        running_avg = sum(running_scores) / len(running_scores) if running_scores else 0

        tally = f"\nRunning Average: {running_avg:.1f}/100 ({len(self.results)} questions)"
        print(tally)
        self.log(tally)

        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log({
                "question_idx": idx,
                "factual_accuracy": scores["factual_accuracy"],
                "completeness": scores["completeness"],
                "technical_precision": scores["technical_precision"],
                "overall_score": overall_score,
                "running_avg": running_avg,
            })

        # Output JSON for streaming (used by backend for WebSocket updates)
        if output_json:
            json_result = {
                "question_idx": idx - 1,  # 0-indexed for frontend
                "question": question,
                "model_response": answer,
                "justification": justification,
                "scores": {
                    "factual_accuracy": scores["factual_accuracy"],
                    "completeness": scores["completeness"],
                    "technical_precision": scores["technical_precision"],
                    "overall_score": overall_score,
                },
                "rag_sources": rag_sources,
            }
            print(json.dumps(json_result), flush=True)

        return result_entry

    def run(self):
        # Write header to log
        self.log("=" * 70)
        self.log("MODEL EVALUATION LOG")
        self.log("=" * 70)
        self.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model: {self.model_name}")
        self.log(f"Dataset: {self.dataset_config.get('dataset_name')}")
        self.log(f"Domain: {self.domain}")
        self.log(f"Judge: {self.judge_model}")
        self.log(f"RAG: Semantic Scholar ({RAG_MAX_PAPERS} papers)")
        self.log(f"Max Samples: {self.max_samples}")
        self.log("=" * 70)

        print()
        print("=" * 70)
        print("STARTING EVALUATION")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.domain}")
        print(f"Judge: {self.judge_model}")
        auth_status = "authenticated" if self.rag.api_key else "unauthenticated"
        print(f"RAG: Semantic Scholar ({RAG_MAX_PAPERS} papers, {auth_status})")
        print("=" * 70)

        # Load data
        test_data = self.load_test_data()
        if not test_data:
            print("No test data!")
            return

        # Evaluate each item with retry logic for failures
        for i, item in enumerate(test_data, 1):
            result = self._evaluate_with_retry(item, i, len(test_data))
            time.sleep(0.2)

        # Finalize and save results
        self._finalize_results()

    def _evaluate_with_retry(self, item: Dict, idx: int, total: int) -> Dict:
        """Evaluate a question with automatic retry on failure."""
        for attempt in range(MAX_RETRIES_ON_ZERO + 1):
            result = self.evaluate_one(item, idx, total, output_json=self.output_json)

            # Determine if we should retry
            should_retry = False
            retry_reason = ""

            # Case 1: No response from model
            if result.get("reason") == "No response":
                should_retry = True
                retry_reason = "no model response"

            # Case 2: Judge model failed
            elif result.get("skipped") and result.get("reason") == "Judge model failed":
                should_retry = True
                retry_reason = "judge model failed"

            # Case 3: No RAG sources found (may succeed on retry with different keywords)
            elif result.get("skipped") and "articles" in result.get("reason", "").lower():
                should_retry = True
                retry_reason = "no RAG sources"

            # Case 4: All-zero scores (parse error or other failure)
            elif not result.get("skipped"):
                scores = result.get("scores", {})
                overall = result.get("overall_score", -1)
                if overall == 0 and all(scores.get(k, 0) == 0 for k in ["factual_accuracy", "completeness", "technical_precision"]):
                    should_retry = True
                    retry_reason = "all-zero scores (parse error)"

            # If no retry needed or max retries reached, return result
            if not should_retry or attempt >= MAX_RETRIES_ON_ZERO:
                if attempt > 0 and not should_retry:
                    print(f"[RETRY SUCCESS] Question {idx} succeeded on attempt {attempt + 1}")
                return result

            # Prepare for retry
            print(f"\n{'!'*70}", flush=True)
            print(f"RETRYING question {idx} (attempt {attempt + 2}/{MAX_RETRIES_ON_ZERO + 1})", flush=True)
            print(f"Reason: {retry_reason}", flush=True)
            print(f"{'!'*70}\n", flush=True)
            self.log(f"[RETRY {attempt + 1}] Question {idx} - {retry_reason}")

            # Remove the failed result before retrying
            if self.results and self.results[-1].get("index") == idx:
                self.results.pop()

            # Also remove from article logs if present
            if self.article_logs and self.article_logs[-1].get("question_index") == idx:
                self.article_logs.pop()

            time.sleep(2.0)  # Longer pause before retry

        return result  # Return last attempt if all retries failed

    def _finalize_results(self):
        """Calculate final results and save to files."""
        # Final results - separate skipped from scored
        # Also exclude zero-score results that failed all retries
        scored_results = [
            r for r in self.results
            if not r.get("skipped", False) and r.get("overall_score", 0) > 0
        ]
        skipped_results = [r for r in self.results if r.get("skipped", False)]
        zero_score_results = [
            r for r in self.results
            if not r.get("skipped", False) and r.get("overall_score", 0) == 0
        ]
        total_scored = len(scored_results)
        total_skipped = len(skipped_results)
        total_zero = len(zero_score_results)
        total_attempted = len(self.results)

        if total_scored == 0:
            print(f"No scored results! ({total_skipped} skipped, {total_zero} failed with zero scores)")
            return

        # Aggregate scores (only from valid scored results)
        avg_factual = sum(r["scores"]["factual_accuracy"] for r in scored_results) / total_scored
        avg_completeness = sum(r["scores"]["completeness"] for r in scored_results) / total_scored
        avg_precision = sum(r["scores"]["technical_precision"] for r in scored_results) / total_scored
        avg_overall = sum(r["overall_score"] for r in scored_results) / total_scored

        final_header = f"\n\n{'='*70}\nFINAL RESULTS\n{'='*70}"
        print(final_header)
        self.log(final_header)

        results_text = f"""
Model:     {self.model_name}
Dataset:   {self.domain}
Judge:     {self.judge_model}
Samples:   {total_scored} scored, {total_skipped} skipped, {total_zero} failed

DIMENSION SCORES (averaged over {total_scored} valid questions):
─────────────────────────────────────
  Factual Accuracy:    {avg_factual:5.1f}/100
  Completeness:        {avg_completeness:5.1f}/100
  Technical Precision: {avg_precision:5.1f}/100
─────────────────────────────────────

  ███ OVERALL ACCURACY: {avg_overall:5.1f}% ███
"""
        print(results_text)
        self.log(results_text)
        self.log("=" * 70)

        # Save JSON
        with open(self.json_path, 'w') as f:
            json.dump({
                "config": {
                    "test_model": self.model_name,
                    "judge_model": self.judge_model,
                    "dataset": self.dataset_config.get('dataset_name'),
                    "domain": self.domain,
                    "samples_attempted": total_attempted,
                    "samples_scored": total_scored,
                    "samples_skipped": total_skipped,
                    "samples_zero": total_zero,
                    "rag_source": "Semantic Scholar API",
                    "rag_papers": RAG_MAX_PAPERS,
                },
                "metrics": {
                    "overall_accuracy": avg_overall,
                    "factual_accuracy": avg_factual,
                    "completeness": avg_completeness,
                    "technical_precision": avg_precision,
                    "total_scored": total_scored,
                    "total_skipped": total_skipped,
                    "total_zero": total_zero,
                },
                "results": self.results,
            }, f, indent=2)

        # Save article logs for manual verification
        with open(self.articles_log_path, 'w') as f:
            json.dump({
                "evaluation_info": {
                    "model": self.model_name,
                    "dataset": self.dataset_config.get('dataset_name'),
                    "domain": self.domain,
                    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_questions": total_attempted,
                    "rag_source": "Semantic Scholar API",
                },
                "article_logs": self.article_logs,
            }, f, indent=2)

        # Log final summary to wandb
        if self.wandb_run:
            self.wandb_run.summary.update({
                "final/overall_accuracy": avg_overall,
                "final/factual_accuracy": avg_factual,
                "final/completeness": avg_completeness,
                "final/technical_precision": avg_precision,
                "final/samples_scored": total_scored,
                "final/samples_skipped": total_skipped,
                "final/samples_failed": total_zero,
            })

        print(f"\nLog saved:     {self.log_path}")
        print(f"JSON saved:    {self.json_path}")
        print(f"Articles log:  {self.articles_log_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments for non-interactive mode."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GGUF models")
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--dataset", type=str, help="Dataset config name (e.g., 'Quantum Physics')")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--judge", type=str, default=JUDGE_MODEL, help="Ollama judge model name")
    parser.add_argument("--output-json", action="store_true", help="Output JSON for parsing")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if running in non-interactive mode (all required args provided)
    non_interactive = args.model and args.dataset

    if not non_interactive:
        print_header()

    # Find GGUF models
    gguf_models = find_gguf_models()
    if not gguf_models and not args.model:
        print("ERROR: No GGUF models found in models/gguf/")
        print("Train a model first, then convert with merge_and_convert_gguff.py")
        sys.exit(1)

    if not non_interactive:
        print(f"Found {len(gguf_models)} GGUF model(s)\n")

    # Select model
    if args.model:
        # Use provided model path
        selected_model_path = Path(args.model)
        selected_model_name = selected_model_path.name
        if not selected_model_path.exists():
            print(f"ERROR: Model file not found: {args.model}")
            sys.exit(1)
    else:
        model_names = [p.name for p in gguf_models]
        selected_model_name = select_from_list(model_names, "Select model to evaluate:")
        selected_model_path = MODELS_DIR / selected_model_name

    if not non_interactive:
        print(f"  -> {selected_model_name}\n")

    # Select dataset
    dataset_configs = load_dataset_configs()
    if not dataset_configs:
        print("ERROR: No dataset configs found in configs/datasets/")
        sys.exit(1)

    if args.dataset:
        # Find matching dataset config (by filename or display name)
        dataset_name = None
        for name, config in dataset_configs.items():
            display_name = config.get('name', name)
            if name.lower() == args.dataset.lower() or display_name.lower() == args.dataset.lower():
                dataset_name = name
                break
        if not dataset_name:
            available = [f"{k} ({v.get('name', k)})" for k, v in dataset_configs.items()]
            print(f"ERROR: Dataset config '{args.dataset}' not found")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        dataset_config = dataset_configs[dataset_name]
    else:
        dataset_names = list(dataset_configs.keys())
        dataset_displays = [f"{name} ({dataset_configs[name].get('dataset_name', '')})"
                            for name in dataset_names]

        print()
        print("Select dataset to test against:")
        print("-" * 50)
        for i, display in enumerate(dataset_displays, 1):
            print(f"  [{i}] {display}")
        print()

        selected_idx = None
        while selected_idx is None:
            try:
                choice = input(f"Select [1-{len(dataset_names)}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(dataset_names):
                    selected_idx = idx
                else:
                    print(f"Please enter 1-{len(dataset_names)}")
            except ValueError:
                print("Enter a number")
            except KeyboardInterrupt:
                print("\nCancelled.")
                sys.exit(0)

        dataset_name = dataset_names[selected_idx]
        dataset_config = dataset_configs[dataset_name]

    if not non_interactive:
        print(f"  -> {dataset_name}\n")

    # Check Ollama for judge model
    judge_model = args.judge
    ollama = OllamaClient()
    judge_models = ollama.list_models()

    # Match judge model (handle :latest suffix)
    matched_judge = None
    for available in judge_models:
        if available == judge_model or available.startswith(f"{judge_model}:"):
            matched_judge = available
            break

    if not matched_judge:
        print(f"ERROR: Judge model '{judge_model}' not found in Ollama.")
        print(f"Available models: {', '.join(judge_models) if judge_models else 'none'}")
        print(f"Pull the model with: ollama pull {judge_model}")
        sys.exit(1)

    judge_model = matched_judge

    if not non_interactive:
        print()
        print(f"Judge model: {judge_model} [OK]")

    # Number of samples
    if args.samples:
        max_samples = args.samples
    else:
        print()
        max_samples = get_number_input("Number of samples to evaluate", default=50, min_val=1, max_val=5000)

    if not non_interactive:
        print(f"  -> {max_samples}\n")

    # Confirm (skip in non-interactive mode)
    if not non_interactive:
        print()
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"Test Model:  {selected_model_name}")
        print(f"Dataset:     {dataset_name}")
        print(f"Judge Model: {judge_model}")
        print(f"RAG:         Semantic Scholar ({RAG_MAX_PAPERS} papers)")
        print(f"Samples:     {max_samples}")
        print("=" * 70)
        print()

        if not get_yes_no("Start evaluation?", default=True):
            print("Cancelled.")
            sys.exit(0)
    else:
        print(f"Starting evaluation: {selected_model_name} on {dataset_name} ({max_samples} samples)")

    # Initialize wandb if requested
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("ERROR: wandb not installed. Run: pip install wandb")
            sys.exit(1)
        dataset_tag = dataset_config.get('domain', dataset_name).lower().replace(' ', '-')
        wandb_run = wandb.init(
            project=f"qwen-fine-tune-test-accuracy-eval-{dataset_tag}",
            config={
                "model": selected_model_name,
                "judge": judge_model,
                "dataset": dataset_config.get('dataset_name'),
                "domain": dataset_config.get('domain', dataset_name),
                "samples": max_samples,
                "rag_source": "Semantic Scholar",
                "rag_papers": RAG_MAX_PAPERS,
            },
        )

    # Load model (use larger context for complex technical questions)
    model = GGUFModel(str(selected_model_path), n_ctx=4096, n_gpu_layers=-1)

    # Run evaluation
    evaluator = Evaluator(
        model=model,
        model_name=selected_model_name,
        judge_client=ollama,
        dataset_config=dataset_config,
        max_samples=max_samples,
        judge_model=judge_model,
        output_json=args.output_json,
        wandb_run=wandb_run,
    )
    evaluator.run()

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
