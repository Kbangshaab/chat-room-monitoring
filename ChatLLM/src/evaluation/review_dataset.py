import json
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ChatLLM/
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_chat_dataset.json"

SEVERITY_LEVELS = ["Ingen", "Lav", "Medium", "Høj"]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_distribution(users: list[dict]) -> None:
    print("=" * 60)
    print(f"Total users: {len(users)}")
    print("=" * 60)

    grimt_counts = Counter(u["grimt_sprog_severity"] for u in users)
    ludomani_counts = Counter(u["ludomani_severity"] for u in users)

    print("\nGrimt sprog distribution:")
    for level in SEVERITY_LEVELS:
        count = grimt_counts.get(level, 0)
        pct = (count / len(users) * 100) if users else 0
        print(f"  {level:8s}: {count:4d}  ({pct:.1f}%)")

    print("\nLudomani distribution:")
    for level in SEVERITY_LEVELS:
        count = ludomani_counts.get(level, 0)
        pct = (count / len(users) * 100) if users else 0
        print(f"  {level:8s}: {count:4d}  ({pct:.1f}%)")

    # Combined cross-tab — useful to spot if certain combinations are missing
    print("\nCombined (grimt x ludomani) cross-tab:")
    print(f"  {'':10s}" + "".join(f"{l:>10s}" for l in SEVERITY_LEVELS))
    for g_level in SEVERITY_LEVELS:
        row = []
        for l_level in SEVERITY_LEVELS:
            count = sum(
                1 for u in users
                if u["grimt_sprog_severity"] == g_level and u["ludomani_severity"] == l_level
            )
            row.append(count)
        print(f"  {g_level:10s}" + "".join(f"{c:>10d}" for c in row))


def flag_suspicious(users: list[dict]) -> list[dict]:
    """Flag users with likely quality issues: too few messages, repeated messages,
    or suspiciously short content."""
    flagged = []

    for u in users:
        messages = u.get("messages", [])
        issues = []

        if len(messages) < 2:
            issues.append("too_few_messages")

        # Check for near-duplicate messages within the same user
        unique_messages = set(m.strip().lower() for m in messages)
        if len(unique_messages) < len(messages):
            issues.append("duplicate_messages")

        # Check for very short messages (likely low-content or broken generation)
        if any(len(m.strip()) < 5 for m in messages):
            issues.append("very_short_message")

        # Very crude repetition check: if any single word appears excessively
        # across all messages combined, flag it (catches the "chikane chikane chikane" case)
        all_words = " ".join(messages).lower().split()
        if all_words:
            word_counts = Counter(all_words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            if len(most_common_word) > 3 and most_common_count >= 4:
                issues.append(f"repeated_word:{most_common_word}({most_common_count}x)")

        if issues:
            flagged.append({**u, "_issues": issues})

    return flagged


def sample_for_manual_review(users: list[dict], n: int = 10, seed: int = 42) -> None:
    """Print a random sample of users for manual eyeballing."""
    rng = random.Random(seed)
    sample = rng.sample(users, min(n, len(users)))

    print("\n" + "=" * 60)
    print(f"RANDOM SAMPLE FOR MANUAL REVIEW (n={len(sample)})")
    print("=" * 60)

    for u in sample:
        print(f"\n--- user_id={u['user_id']} | grimt={u['grimt_sprog_severity']} | ludomani={u['ludomani_severity']} ---")
        for m in u["messages"]:
            print(f"  - {m}")


if __name__ == "__main__":
    users = load_dataset()

    print_distribution(users)

    flagged = flag_suspicious(users)
    print(f"\n{'=' * 60}")
    print(f"FLAGGED FOR REVIEW: {len(flagged)} / {len(users)} users ({len(flagged)/len(users)*100:.1f}%)")
    print("=" * 60)
    for u in flagged[:20]:  # cap printed output, full list is in the variable if needed
        print(f"\nuser_id={u['user_id']} | issues={u['_issues']}")
        for m in u["messages"]:
            print(f"  - {m}")

    sample_for_manual_review(users, n=10)