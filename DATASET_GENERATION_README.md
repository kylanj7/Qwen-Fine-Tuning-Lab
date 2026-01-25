# Dataset Generation Agent

This tool automatically generates instruction-following datasets from scholarly articles for use with the fine-tuning pipeline.

## Features

- **Multi-format Support**: Reads PDF, TXT, MD, and other text formats
- **Automatic Extraction**: Extracts instruction-response pairs using pattern matching
- **Domain Classification**: Automatically classifies domain and sub-domain
- **Training-Ready Format**: Outputs datasets in the exact format required by your training pipeline
- **Flexible Output**: Supports JSON, Parquet, and HuggingFace dataset formats

## Installation

```bash
# Install dataset generation dependencies
pip install -r requirements_dataset_gen.txt

# Or install manually
pip install PyPDF2 datasets pandas
```

## Quick Start

### Basic Usage

```bash
# Process a single PDF
python generate_dataset_from_articles.py \
    --input paper.pdf \
    --output my_dataset.json

# Process all articles in a directory
python generate_dataset_from_articles.py \
    --input articles/ \
    --output my_dataset.json

# Specify domain manually
python generate_dataset_from_articles.py \
    --input paper.pdf \
    --output quantum_dataset.json \
    --domain "Quantum Physics" \
    --sub-domain "Quantum Computing"
```

### Advanced Usage

```bash
# Use LLM for better extraction (optional)
python generate_dataset_from_articles.py \
    --input articles/ \
    --output dataset.json \
    --use-llm \
    --llm-model "Qwen/Qwen2.5-14B-Instruct"

# Export as HuggingFace dataset format
python generate_dataset_from_articles.py \
    --input articles/ \
    --output my_dataset/ \
    --format huggingface
```

## Dataset Format

The generated dataset matches your training pipeline format:

### Chemistry Format
```json
{
  "message_1": "What is a covalent bond?",
  "message_2": "A covalent bond is a chemical bond...",
  "topic;": "Chemistry",
  "sub_topic": "Chemical Bonding"
}
```

### Quantum Format (also included)
```json
{
  "problem": "What is a covalent bond?",
  "solution": "A covalent bond is a chemical bond...",
  "main_domain": "Chemistry",
  "sub_domain": "Chemical Bonding"
}
```

## Extraction Methods

The tool uses multiple pattern-matching strategies:

1. **Q&A Patterns**: Extracts "Q: ... A: ..." or "Question: ... Answer: ..."
2. **Section Headers**: Uses section headers as questions, content as answers
3. **Definition Patterns**: Finds "What is X?" questions with answers
4. **LLM Extraction** (optional): Uses language model for more sophisticated extraction

## Domain Classification

Automatic domain classification supports:

- **Quantum Physics**
  - Quantum Computing
  - Quantum Mechanics
  - Quantum Information

- **Chemistry**
  - Organic Chemistry
  - Physical Chemistry
  - Inorganic Chemistry

- **Physics**
  - Classical Mechanics
  - Electromagnetism
  - Thermodynamics

## Using Generated Datasets

After generating a dataset, use it with your training pipeline:

```python
from datasets import load_dataset
import json

# Load generated dataset
with open('my_dataset.json', 'r') as f:
    data = json.load(f)

# Or use the helper script
from example_use_generated_dataset import load_generated_dataset, format_dataset_for_training

dataset = load_generated_dataset('my_dataset.json')
formatted = format_dataset_for_training(dataset, dataset_type="chemistry")

# Use with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted,
    ...
)
```

## Integration with Training Pipeline

Modify your training script to load generated datasets:

```python
# In chemistry_dataset_train.py or train.py

# Option 1: Load from JSON
import json
with open('my_generated_dataset.json', 'r') as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# Option 2: Load from HuggingFace format
from datasets import load_dataset
dataset = load_dataset('path/to/my_generated_dataset', split='train')

# Apply your existing formatting function
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Tips for Better Results

1. **Pre-process Articles**: Clean PDFs often extract better than scanned documents
2. **Domain Specification**: Manually specify domain if auto-classification is incorrect
3. **Review Output**: Check generated dataset and adjust extraction patterns if needed
4. **Combine Sources**: Process multiple articles to build larger datasets
5. **LLM Extraction**: Use `--use-llm` for more sophisticated extraction (slower but better)

## Troubleshooting

### No entries extracted
- Check that your articles contain Q&A patterns or section headers
- Try manually specifying domain with `--domain` and `--sub-domain`
- Consider using `--use-llm` for better extraction

### PDF parsing errors
- Ensure PyPDF2 is installed: `pip install PyPDF2`
- Some PDFs may be scanned images (OCR needed)
- Try converting PDF to text first

### Format errors
- Ensure output path has correct extension (.json, .parquet, or directory for HuggingFace)
- Check that datasets library is installed for Parquet/HuggingFace formats

## Examples

### Example 1: Process Research Papers
```bash
# Download papers to articles/ directory
python generate_dataset_from_articles.py \
    --input articles/ \
    --output research_dataset.json \
    --domain "Quantum Physics"
```

### Example 2: Chemistry Textbook
```bash
python generate_dataset_from_articles.py \
    --input chemistry_textbook.pdf \
    --output chemistry_dataset.json \
    --domain "Chemistry" \
    --sub-domain "General Chemistry"
```

### Example 3: Multiple Domains
```bash
# Process quantum papers
python generate_dataset_from_articles.py \
    --input quantum_papers/ \
    --output quantum_dataset.json \
    --domain "Quantum Physics"

# Process chemistry papers
python generate_dataset_from_articles.py \
    --input chemistry_papers/ \
    --output chemistry_dataset.json \
    --domain "Chemistry"

# Combine datasets (in Python)
from datasets import load_dataset, concatenate_datasets
quantum = load_dataset('json', data_files='quantum_dataset.json')['train']
chemistry = load_dataset('json', data_files='chemistry_dataset.json')['train']
combined = concatenate_datasets([quantum, chemistry])
combined.to_json('combined_dataset.json')
```

## Output Statistics

The tool prints statistics during processing:
- Number of characters extracted
- Number of instruction-response pairs found
- Domain classifications
- Final entry count

## Next Steps

1. Generate your dataset: `python generate_dataset_from_articles.py --input articles/ --output dataset.json`
2. Review the generated dataset
3. Use with training pipeline: Modify `train.py` or `chemistry_dataset_train.py` to load your dataset
4. Fine-tune your model!

---

For questions or issues, check the extraction patterns in `generate_dataset_from_articles.py` and adjust as needed for your specific article formats.
