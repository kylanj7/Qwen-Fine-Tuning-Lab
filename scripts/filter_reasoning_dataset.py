"""
Filter and validate reasoning dataset for SFT quality.

Features:
- Validates <think>/<answer> tag structure
- HEALER: Recovers broken samples by extracting \\boxed{} answers
- Fixes common formatting issues
- Removes truly unrecoverable samples
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Validation patterns
THINK_OPEN = re.compile(r'<think>', re.IGNORECASE)
THINK_CLOSE = re.compile(r'</think>', re.IGNORECASE)
ANSWER_OPEN = re.compile(r'<answer>', re.IGNORECASE)
ANSWER_CLOSE = re.compile(r'</answer>', re.IGNORECASE)

# Healer patterns - for extracting answers from broken responses
BOXED_PATTERN = re.compile(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)
FINAL_ANSWER_PATTERNS = [
    re.compile(r'(?:final\s+)?(?:answer|result)\s*(?:is|:)\s*["\']?([^"\'.\n]+)', re.IGNORECASE),
    re.compile(r'(?:therefore|thus|hence)[,\s]+([^.\n]+)', re.IGNORECASE),
    re.compile(r'=\s*([^=\n]+)$', re.MULTILINE),
]

# LaTeX patterns - detect inconsistent formatting
LATEX_INLINE = re.compile(r'\$[^$]+\$')  # $...$
LATEX_DISPLAY = re.compile(r'\\\[.+?\\\]', re.DOTALL)  # \[...\]
LATEX_OPERATORS = re.compile(r'\\(sigma|psi|phi|omega|alpha|beta|gamma|delta|epsilon|hbar|nabla|partial|frac|sqrt|sum|prod|int|ket|bra|langle|rangle)')

# Bad patterns - plain text physics that should be LaTeX
PLAIN_TEXT_PHYSICS = [
    r'\bsigma_[xyz]\b',  # sigma_x instead of \sigma_x
    r'\bpsi\b(?![_^\\])',  # psi without LaTeX
    r'\bH\s*=\s*[^$\\]',  # H = without LaTeX
    r'\bhbar\b(?!\\)',  # hbar without backslash
]


# =============================================================================
# HEALER FUNCTIONS - Recover broken but salvageable samples
# =============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{} LaTeX command."""
    matches = BOXED_PATTERN.findall(text)
    if matches:
        # Return the last boxed result (usually the final answer)
        return matches[-1].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Try to extract final answer from conversational patterns."""
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            answer = match.group(1).strip()
            # Clean up common trailing patterns
            answer = re.sub(r'\s*[.,:;]?\s*$', '', answer)
            if len(answer) > 3:  # Avoid single characters
                return answer
    return None


def heal_answer_tag(output: str) -> Tuple[str, bool, str]:
    """
    Attempt to heal a broken <answer> tag.

    Returns:
        Tuple of (healed_output, was_healed, heal_method)
    """
    # Check if already valid
    if ANSWER_OPEN.search(output) and ANSWER_CLOSE.search(output):
        # Check if answer content is empty or just whitespace
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)
        if answer_match and answer_match.group(1).strip():
            return output, False, "already_valid"

    # Strategy 1 (PRIORITY): Missing </answer> tag with existing content
    # This is the most common failure mode - just add the closing tag
    if ANSWER_OPEN.search(output) and not ANSWER_CLOSE.search(output):
        # Check if there's actual content after <answer>
        answer_start = re.search(r'<answer>', output, re.IGNORECASE)
        if answer_start:
            content_after = output[answer_start.end():].strip()
            if len(content_after) > 20:  # Has substantial content
                healed = output.rstrip() + '\n</answer>'
                return healed, True, "close_tag"

    # Strategy 2: No answer tags - extract from \boxed{} anywhere in the text
    if not ANSWER_OPEN.search(output):
        boxed = extract_boxed_answer(output)
        if boxed and THINK_CLOSE.search(output):
            # Insert proper answer block after </think>
            replacement = f'</think>\n\n<answer>\n{boxed}\n</answer>'
            healed = THINK_CLOSE.sub(lambda m: replacement, output, count=1)
            return healed, True, "boxed_extraction"

    # Strategy 3: No answer tags - extract from conversational patterns
    if not ANSWER_OPEN.search(output):
        extracted = extract_final_answer(output)
        if extracted and THINK_CLOSE.search(output):
            replacement = f'</think>\n\n<answer>\n{extracted}\n</answer>'
            healed = THINK_CLOSE.sub(lambda m: replacement, output, count=1)
            return healed, True, "pattern_extraction"

    # Strategy 4: Answer tag exists but content is empty - inject boxed answer
    if ANSWER_OPEN.search(output):
        answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', output, re.DOTALL | re.IGNORECASE)
        if answer_match and len(answer_match.group(1).strip()) < 20:
            boxed = extract_boxed_answer(output)
            if boxed:
                replacement = f'<answer>\n{boxed}\n</answer>'
                healed = re.sub(
                    r'<answer>.*?(?:</answer>|$)',
                    lambda m: replacement,
                    output,
                    flags=re.DOTALL | re.IGNORECASE
                )
                return healed, True, "answer_replacement"

    # Strategy 5: Fallback - just add closing tag if answer open exists
    if ANSWER_OPEN.search(output) and not ANSWER_CLOSE.search(output):
        healed = output.rstrip() + '\n</answer>'
        return healed, True, "close_tag_fallback"

    return output, False, "unrecoverable"


def heal_think_tag(output: str) -> Tuple[str, bool, str]:
    """Attempt to heal broken <think> tags."""
    has_open = THINK_OPEN.search(output)
    has_close = THINK_CLOSE.search(output)

    if has_open and has_close:
        return output, False, "already_valid"

    # Missing opening tag - add at start
    if not has_open and has_close:
        healed = '<think>\n' + output
        return healed, True, "add_open_tag"

    # Missing closing tag - add before <answer> or at reasonable point
    if has_open and not has_close:
        if ANSWER_OPEN.search(output):
            healed = ANSWER_OPEN.sub(lambda m: '</think>\n\n<answer>', output, count=1)
            return healed, True, "add_close_before_answer"
        else:
            # Try to find end of reasoning (look for conclusion patterns)
            conclusion_match = re.search(r'(therefore|thus|hence|finally|in conclusion)[^.]*\.', output, re.IGNORECASE)
            if conclusion_match:
                pos = conclusion_match.end()
                healed = output[:pos] + '\n</think>\n' + output[pos:]
                return healed, True, "add_close_at_conclusion"

    return output, False, "unrecoverable"


def heal_sample(output: str) -> Tuple[str, bool, List[str]]:
    """
    Attempt to heal all issues in a sample.

    Returns:
        Tuple of (healed_output, was_healed, list_of_methods_used)
    """
    methods = []
    healed = output
    any_healed = False

    # First heal think tags
    healed, think_healed, think_method = heal_think_tag(healed)
    if think_healed:
        methods.append(f"think:{think_method}")
        any_healed = True

    # Then heal answer tags
    healed, answer_healed, answer_method = heal_answer_tag(healed)
    if answer_healed:
        methods.append(f"answer:{answer_method}")
        any_healed = True

    return healed, any_healed, methods


def validate_tags(output: str) -> Tuple[bool, List[str]]:
    """Validate <think> and <answer> tags are properly opened and closed."""
    issues = []

    think_opens = len(THINK_OPEN.findall(output))
    think_closes = len(THINK_CLOSE.findall(output))
    answer_opens = len(ANSWER_OPEN.findall(output))
    answer_closes = len(ANSWER_CLOSE.findall(output))

    if think_opens == 0:
        issues.append("Missing <think> tag")
    if think_closes == 0:
        issues.append("Missing </think> tag")
    if answer_opens == 0:
        issues.append("Missing <answer> tag")
    if answer_closes == 0:
        issues.append("Missing </answer> tag - CRITICAL: model learned to never conclude")

    if think_opens != think_closes:
        issues.append(f"Unbalanced think tags: {think_opens} opens, {think_closes} closes")
    if answer_opens != answer_closes:
        issues.append(f"Unbalanced answer tags: {answer_opens} opens, {answer_closes} closes")

    # Check tag order: <think>...</think><answer>...</answer>
    if not issues:
        think_start = output.lower().find('<think>')
        think_end = output.lower().find('</think>')
        answer_start = output.lower().find('<answer>')
        answer_end = output.lower().find('</answer>')

        if not (think_start < think_end < answer_start < answer_end):
            issues.append("Tags not in correct order: <think>...</think><answer>...</answer>")

    return len(issues) == 0, issues


def strip_math_environments(text: str) -> str:
    """Remove content inside LaTeX math environments to avoid false positives."""
    # Remove display math: \[...\]
    text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
    # Remove inline math: $...$  (but not escaped \$)
    text = re.sub(r'(?<!\\)\$[^$]+(?<!\\)\$', ' ', text)
    # Remove \(...\) inline math
    text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
    # Remove equation environments
    text = re.sub(r'\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}', ' ', text, flags=re.DOTALL)
    return text


def validate_latex(output: str) -> Tuple[bool, List[str]]:
    """Check for LaTeX consistency."""
    issues = []

    # Strip math environments before checking for plain text physics
    text_outside_math = strip_math_environments(output)

    # Check for plain text physics that should be LaTeX (only outside math mode)
    for pattern in PLAIN_TEXT_PHYSICS:
        if re.search(pattern, text_outside_math):
            issues.append(f"Plain text physics found (should be LaTeX): {pattern}")

    # Check that LaTeX is used
    has_latex = bool(LATEX_INLINE.search(output) or LATEX_DISPLAY.search(output) or LATEX_OPERATORS.search(output))
    if not has_latex:
        issues.append("No LaTeX math detected - physics response should contain LaTeX")

    # Check for unclosed LaTeX
    dollar_count = output.count('$') - output.count('\\$')  # Exclude escaped
    if dollar_count % 2 != 0:
        issues.append("Unclosed LaTeX $ delimiter")

    return len(issues) == 0, issues


def validate_reasoning_quality(output: str) -> Tuple[bool, List[str]]:
    """Check for reasoning quality issues."""
    issues = []

    # Extract think content
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)

        # Check for minimum reasoning length
        if len(think_content.strip()) < 100:
            issues.append("Reasoning too short (< 100 chars) - may be superficial")

        # Check for step-by-step indicators
        has_steps = any(marker in think_content.lower() for marker in
                       ['step 1', 'first,', '1.', '1)', 'let us', 'we begin', 'starting with'])
        if not has_steps:
            issues.append("No clear step-by-step structure detected")

    # Extract answer content
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_content = answer_match.group(1)

        if len(answer_content.strip()) < 10:
            issues.append("Answer too short - may be incomplete")

    # Check for reasoning loops (repetitive phrases)
    sentences = output.split('.')
    if len(sentences) > 5:
        unique_starts = set(s.strip()[:30].lower() for s in sentences if len(s.strip()) > 30)
        if len(unique_starts) < len(sentences) * 0.5:
            issues.append("Possible reasoning loop - repetitive sentence structures")

    return len(issues) == 0, issues


def filter_dataset(input_file: str, output_file: str = None, strict: bool = True, heal: bool = True):
    """
    Filter a JSONL dataset, attempting to heal broken samples before discarding.

    Args:
        input_file: Path to input JSONL
        output_file: Path to output JSONL (default: input_filtered.jsonl)
        strict: If True, discard samples with any issues. If False, only discard critical issues.
        heal: If True, attempt to heal broken samples before validation.
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = input_path.stem + "_filtered.jsonl"

    stats = {
        "total": 0,
        "passed_original": 0,
        "passed_healed": 0,
        "failed_tags": 0,
        "failed_latex": 0,
        "failed_quality": 0,
        "heal_attempts": 0,
        "heal_success": 0,
        "issues": []
    }

    valid_samples = []

    print(f"Filtering: {input_file}")
    print(f"Mode: {'Strict' if strict else 'Lenient'} | Healing: {'ON' if heal else 'OFF'}")
    print("=" * 60)

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1

            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError:
                stats["issues"].append((line_num, "Invalid JSON"))
                continue

            output = sample.get("output", "")
            original_output = output

            # HEALING PHASE: Attempt to fix broken samples
            healed = False
            heal_methods = []
            if heal:
                # First check if healing is needed
                tags_ok, _ = validate_tags(output)
                if not tags_ok:
                    stats["heal_attempts"] += 1
                    output, healed, heal_methods = heal_sample(output)
                    if healed:
                        stats["heal_success"] += 1
                        sample["output"] = output
                        sample["healed"] = True
                        sample["heal_methods"] = heal_methods

            # VALIDATION PHASE
            all_issues = []
            critical_fail = False

            # Validate tags (CRITICAL)
            tags_ok, tag_issues = validate_tags(output)
            if not tags_ok:
                stats["failed_tags"] += 1
                all_issues.extend(tag_issues)
                # Missing </answer> is critical - always discard
                if any("</answer>" in issue for issue in tag_issues):
                    critical_fail = True

            # Validate LaTeX
            latex_ok, latex_issues = validate_latex(output)
            if not latex_ok:
                stats["failed_latex"] += 1
                all_issues.extend(latex_issues)

            # Validate reasoning quality
            quality_ok, quality_issues = validate_reasoning_quality(output)
            if not quality_ok:
                stats["failed_quality"] += 1
                all_issues.extend(quality_issues)

            # Decide whether to keep
            if strict:
                keep = len(all_issues) == 0
            else:
                keep = not critical_fail

            if keep:
                if healed:
                    stats["passed_healed"] += 1
                else:
                    stats["passed_original"] += 1
                valid_samples.append(sample)
            else:
                stats["issues"].append((line_num, all_issues, healed, heal_methods))
                if stats["total"] <= 20:  # Show first 20 failures
                    heal_info = f" (healed: {heal_methods})" if healed else ""
                    print(f"[{line_num}] REJECTED{heal_info}: {all_issues[0] if all_issues else 'Unknown'}")

    # Write filtered output
    with open(output_file, 'w') as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    total_passed = stats['passed_original'] + stats['passed_healed']

    # Report
    print("=" * 60)
    print(f"FILTER RESULTS")
    print("=" * 60)
    print(f"Total samples:     {stats['total']}")
    print(f"Passed (total):    {total_passed} ({100*total_passed/stats['total']:.1f}%)")
    print(f"  - Original:      {stats['passed_original']}")
    print(f"  - Healed:        {stats['passed_healed']}")
    if heal:
        print(f"\nHealing stats:")
        print(f"  - Attempts:      {stats['heal_attempts']}")
        print(f"  - Successful:    {stats['heal_success']} ({100*stats['heal_success']/max(1,stats['heal_attempts']):.1f}%)")
    print(f"\nRejected:          {stats['total'] - total_passed}")
    print(f"  - Tag issues:    {stats['failed_tags']}")
    print(f"  - LaTeX issues:  {stats['failed_latex']}")
    print(f"  - Quality issues:{stats['failed_quality']}")
    print(f"\nFiltered output:   {output_file}")

    # Show sample issues
    if stats["issues"][:5]:
        print(f"\nSample rejection details (first 5):")
        for item in stats["issues"][:5]:
            if len(item) == 4:
                line_num, issues, healed, methods = item
                heal_info = f" [healed: {methods}]" if healed else ""
                print(f"  Line {line_num}{heal_info}: {issues[:2]}...")
            else:
                line_num, issues = item
                print(f"  Line {line_num}: {issues}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter reasoning dataset for SFT quality")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file")
    parser.add_argument("--lenient", action="store_true", help="Only reject critical failures (missing answer tags)")

    args = parser.parse_args()

    filter_dataset(args.input, args.output, strict=not args.lenient)
