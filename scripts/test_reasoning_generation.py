"""Test reasoning generation with a small sample."""
import openai
import json
import time
import re
from datasets import load_dataset

# Setup Local Ollama Client
client = openai.OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

MODEL_NAME = "gpt-oss:120b"
NUM_SAMPLES = 5
NUM_CTX = 16384  # Context window - reasoning models need large scratchpad

# Strict system prompt (same as generate_reasoning_dataset.py)
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


def validate_output(content: str) -> dict:
    """Comprehensive validation of generated output."""
    lower = content.lower()

    result = {
        # Tag validation
        "has_think_open": "<think>" in lower,
        "has_think_close": "</think>" in lower,
        "has_answer_open": "<answer>" in lower,
        "has_answer_close": "</answer>" in lower,
        # Math validation
        "has_latex": bool(re.search(r'\\[a-zA-Z]+|\\\[|\$', content)),
        # Structure validation
        "has_steps": bool(re.search(r'step\s*\d|step\s*[1-4]:', lower)),
        # First Principles checks
        "has_hilbert_space": bool(re.search(r'hilbert|dimension|\\mathbb\{C\}|\$\\mathcal\{H\}', lower)),
        "has_hamiltonian": bool(re.search(r'hamiltonian|\\hat\{H\}|\$H\s*=', content)),
        "has_normalization": bool(re.search(r'normali[zs]|unitar|\\langle.*\\rangle\s*=\s*1', lower)),
        "has_sanity_check": bool(re.search(r'sanity|check|verify|limit|recover|consistent', lower)),
    }

    result["all_tags_valid"] = all([
        result["has_think_open"],
        result["has_think_close"],
        result["has_answer_open"],
        result["has_answer_close"]
    ])

    # Count First Principles adherence
    first_principles = ["has_hilbert_space", "has_hamiltonian", "has_normalization", "has_sanity_check"]
    result["first_principles_score"] = sum(1 for fp in first_principles if result[fp])

    return result


def generate_reasoning_path(question, answer):
    user_msg = f"""Question: {question}

Reference Answer: {answer}

Generate a complete step-by-step reasoning path leading to this answer."""

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
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"Loading dataset...")
    ds = load_dataset("BoltzmannEntropy/QuantumLLMInstruct", split="train")

    print(f"Generating {NUM_SAMPLES} test samples with {MODEL_NAME}...")
    print(f"Using strict prompt with LaTeX and self-verification requirements\n")
    print("=" * 80)

    results = []
    for i in range(NUM_SAMPLES):
        item = ds[i]
        q = item["problem"]
        a = item["solution"]

        print(f"\n[{i+1}/{NUM_SAMPLES}] QUESTION:")
        print(q[:200] + "..." if len(q) > 200 else q)
        print()

        start = time.time()
        output = generate_reasoning_path(q, a)
        elapsed = time.time() - start

        if output:
            print(f"GENERATED ({elapsed:.1f}s):")
            print("-" * 40)
            # Show full output for review
            print(output)
            print("-" * 40)

            # Detailed validation
            validation = validate_output(output)
            print(f"\nVALIDATION:")
            print(f"  Tags:          {'✓' if validation['all_tags_valid'] else '✗'} (think: {validation['has_think_open']}/{validation['has_think_close']}, answer: {validation['has_answer_open']}/{validation['has_answer_close']})")
            print(f"  LaTeX:         {'✓' if validation['has_latex'] else '✗'}")
            print(f"  Step format:   {'✓' if validation['has_steps'] else '✗'}")
            print(f"\n  FIRST PRINCIPLES ({validation['first_principles_score']}/4):")
            print(f"    Hilbert space: {'✓' if validation['has_hilbert_space'] else '✗'}")
            print(f"    Hamiltonian:   {'✓' if validation['has_hamiltonian'] else '✗'}")
            print(f"    Normalization: {'✓' if validation['has_normalization'] else '✗'}")
            print(f"    Sanity check:  {'✓' if validation['has_sanity_check'] else '✗'}")

            # Quality assessment
            if validation['all_tags_valid'] and validation['has_latex'] and validation['first_principles_score'] >= 2:
                quality = "PASS"
            elif validation['all_tags_valid'] and validation['has_latex']:
                quality = "MARGINAL"
            else:
                quality = "FAIL"
            print(f"\n  >>> {quality}")

            results.append({
                "question": q,
                "original_answer": a,
                "generated": output,
                "validation": validation,
                "quality": quality,
                "time": elapsed
            })
        else:
            print("FAILED to generate")
            results.append({
                "question": q,
                "original_answer": a,
                "generated": None,
                "quality": "ERROR",
                "time": elapsed
            })

        print("=" * 80)

    # Summary
    passed = sum(1 for r in results if r.get("quality") == "PASS")
    marginal = sum(1 for r in results if r.get("quality") == "MARGINAL")
    failed = sum(1 for r in results if r.get("quality") == "FAIL")
    errors = sum(1 for r in results if r.get("quality") == "ERROR")
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0

    # Calculate average first principles score
    fp_scores = [r.get("validation", {}).get("first_principles_score", 0) for r in results if r.get("validation")]
    avg_fp = sum(fp_scores) / len(fp_scores) if fp_scores else 0

    print(f"\n{'=' * 80}")
    print(f"QUALITY SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples:       {NUM_SAMPLES}")
    print(f"PASS:                {passed} ({100*passed/NUM_SAMPLES:.0f}%)")
    print(f"MARGINAL:            {marginal} ({100*marginal/NUM_SAMPLES:.0f}%)")
    print(f"FAIL:                {failed} ({100*failed/NUM_SAMPLES:.0f}%)")
    print(f"ERROR:               {errors}")
    print(f"\nFirst Principles avg: {avg_fp:.1f}/4")
    print(f"Avg time/sample:     {avg_time:.1f}s")
    print(f"Est. full dataset:   {(5150 * avg_time / 3600):.1f} hours")

    usable = passed + marginal
    if usable / NUM_SAMPLES >= 0.8:
        print(f"\n✓ Quality looks good! {usable}/{NUM_SAMPLES} usable samples.")
        print(f"  Ready for full generation.")
    else:
        print(f"\n⚠ Only {usable}/{NUM_SAMPLES} usable. Consider:")
        print(f"  - Adjusting the system prompt")
        print(f"  - Using a different model")
        print(f"  - Post-filtering with filter_reasoning_dataset.py")

    # Save test results
    with open("test_reasoning_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: test_reasoning_output.json")

if __name__ == "__main__":
    main()
