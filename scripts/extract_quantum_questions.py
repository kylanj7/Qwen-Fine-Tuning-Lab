"""Extract quantum physics questions for reasoning dataset generation."""
from datasets import load_dataset
import json

# Load the quantum physics dataset from HuggingFace
print("Loading dataset from HuggingFace...")
ds = load_dataset("BoltzmannEntropy/QuantumLLMInstruct", split="train")

print(f"Total examples: {len(ds)}")

# Extract question/answer pairs
output = []
for item in ds:
    output.append({
        "question": item["problem"],
        "answer": item["solution"],
        "domain": item.get("main_domain", ""),
        "sub_domain": item.get("sub_domain", "")
    })

# Save to JSON
output_file = "quantum_basics.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(output)} examples to {output_file}")

# Also create a smaller sample for testing
sample_file = "quantum_basics_sample.json"
with open(sample_file, 'w') as f:
    json.dump(output[:50], f, indent=2)

print(f"Saved 50-example sample to {sample_file}")
