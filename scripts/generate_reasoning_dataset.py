"""Generate reasoning dataset for quantum physics fine-tuning."""
import openai
import json
import time
import re
from pathlib import Path

# 1. Setup Local Ollama Client
client = openai.OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

# 2. Configuration
MODEL_NAME = "gpt-oss:120b"
INPUT_FILE = "quantum_basics.json"  # Run extract_quantum_questions.py first
OUTPUT_FILE = "../datasets/quantum_reasoning_v1.jsonl"
MAX_RETRIES = 3  # Retries per question on format failure
NUM_CTX = 16384  # Context window - reasoning models need large scratchpad

# Strict system prompt for consistent, high-quality reasoning
SYSTEM_PROMPT = """You are an expert Quantum Physicist and Logic Engine. Your goal is to solve complex quantum mechanics and quantum computing problems using a rigorous, step-by-step reasoning process.

### STRICTURES:
1. OUTPUT FORMAT: You MUST wrap your internal derivation inside <think> tags and provide the final, concise solution inside <answer> tags.
2. MATHEMATICAL RIGOR: Use LaTeX for ALL mathematical expressions. Never use plain text for variables (e.g., use $\\psi$, not 'psi').
3. THE "FIRST PRINCIPLES" RULE: In the <think> section, always:
    - Define the Hilbert space dimension.
    - Write out the full Hamiltonian or Density Matrix explicitly before simplifying.
    - Check for normalization/unitarity at each step.
    - If a calculation is complex, perform a "sanity check" (e.g., "Does this limit recover the ground state?").
4. COMPLETENESS: Never truncate your response. If the calculation is long, prioritize mathematical steps over conversational filler.
5. NO OMISSIONS: Do not skip steps with "it follows that" unless you have shown the algebraic substitution immediately prior.
6. LENGTH LIMIT: If your derivation exceeds 1,000 words, summarize the key mathematical steps and move immediately to the <answer> block. Do not loop or repeat reasoning.

### RESPONSE TEMPLATE:
<think>
[Step 1: Problem Decomposition]
[Step 2: Mathematical Setup/Operators]
[Step 3: Intermediate Calculations/Derivations]
[Step 4: Verification/Sanity Check]
</think>

<answer>
[Final concise result with units and core physical interpretation]
</answer>

CRITICAL: Responses with missing </answer> tags or plain-text math are REJECTED."""


def validate_output(content: str) -> tuple[bool, str]:
    """Validate the generated output has correct format."""
    issues = []

    # Check for required tags
    if "<think>" not in content.lower():
        issues.append("missing <think>")
    if "</think>" not in content.lower():
        issues.append("missing </think>")
    if "<answer>" not in content.lower():
        issues.append("missing <answer>")
    if "</answer>" not in content.lower():
        issues.append("missing </answer> - CRITICAL")

    # Check tag order
    if not issues:
        lower = content.lower()
        positions = [
            lower.find("<think>"),
            lower.find("</think>"),
            lower.find("<answer>"),
            lower.find("</answer>")
        ]
        if positions != sorted(positions):
            issues.append("tags out of order")

    # Check for LaTeX
    has_latex = bool(re.search(r'\\[a-zA-Z]+|\\\[|\$', content))
    if not has_latex:
        issues.append("no LaTeX detected")

    return len(issues) == 0, ", ".join(issues) if issues else "valid"


def generate_reasoning_path(question, answer, attempt=1):
    """Generate reasoning with validation and retry."""
    user_msg = f"""Question: {question}

Reference Answer: {answer}

Generate a complete step-by-step reasoning path leading to this answer. Remember:
- Use <think>...</think> for reasoning
- Use <answer>...</answer> for the final result
- ALL math must be in LaTeX format
- You MUST close both tags"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            extra_body={"num_ctx": NUM_CTX},  # Ollama context window
        )
        content = response.choices[0].message.content

        # Validate output
        is_valid, validation_msg = validate_output(content)

        if is_valid:
            return content, True
        else:
            print(f"  [Attempt {attempt}] Invalid: {validation_msg}")
            return content, False

    except Exception as e:
        print(f"  [Attempt {attempt}] Error: {e}")
        return None, False

def main():
    # Check input file exists
    if not Path(INPUT_FILE).exists():
        print(f"Error: {INPUT_FILE} not found!")
        print("Run extract_quantum_questions.py first to create it.")
        return

    # Load input data
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)

    print(f"Processing {len(raw_data)} examples...")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Max retries per question: {MAX_RETRIES}")
    print("=" * 60)

    # Ensure output directory exists
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Track stats
    stats = {
        "processed": 0,
        "valid": 0,
        "invalid_kept": 0,
        "skipped": 0,
        "total_retries": 0
    }

    with open(OUTPUT_FILE, 'a') as out_f:
        for i, entry in enumerate(raw_data):
            q = entry['question']
            a = entry['answer']

            print(f"\n[{i+1}/{len(raw_data)}] {q[:60]}...")

            best_output = None
            is_valid = False

            # Try up to MAX_RETRIES times
            for attempt in range(1, MAX_RETRIES + 1):
                output, valid = generate_reasoning_path(q, a, attempt)

                if output:
                    best_output = output
                    is_valid = valid

                if valid:
                    break  # Success, no need to retry

                if attempt < MAX_RETRIES:
                    stats["total_retries"] += 1
                    time.sleep(1)  # Brief pause before retry

            # Decide what to do with the result
            if best_output is None:
                print(f"  SKIPPED: No output generated")
                stats["skipped"] += 1
                continue

            if is_valid:
                print(f"  ✓ Valid output")
                stats["valid"] += 1
            else:
                # Keep invalid output but log it - filter script will handle later
                print(f"  ⚠ Kept invalid output (will filter later)")
                stats["invalid_kept"] += 1

            # Write to output
            jsonl_line = {
                "instruction": q,
                "output": best_output,
                "valid": is_valid  # Tag for filtering
            }
            out_f.write(json.dumps(jsonl_line) + "\n")
            stats["processed"] += 1

            # Progress update every 25 samples
            if stats["processed"] % 25 == 0:
                valid_pct = 100 * stats["valid"] / stats["processed"] if stats["processed"] > 0 else 0
                print(f"\n--- Progress: {stats['processed']}/{len(raw_data)} | Valid: {valid_pct:.1f}% ---\n")

    # Final report
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total processed:   {stats['processed']}")
    print(f"Valid outputs:     {stats['valid']} ({100*stats['valid']/stats['processed']:.1f}%)" if stats['processed'] > 0 else "")
    print(f"Invalid (kept):    {stats['invalid_kept']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"Total retries:     {stats['total_retries']}")
    print(f"\nOutput saved to:   {OUTPUT_FILE}")
    print(f"\nNext step: Run filter script to clean invalid samples:")
    print(f"  python filter_reasoning_dataset.py {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
