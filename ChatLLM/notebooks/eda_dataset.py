import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path.cwd().parents[0] if "notebooks" in str(Path.cwd()) else Path.cwd()
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_chat_dataset.json"
(PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["Ingen", "Lav", "Medium", "Høj"]

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    users = json.load(f)

print(f"Loaded {len(users)} users")

# %% [markdown]
# ## 1. Convert to DataFrame for easier analysis

# %%
df = pd.DataFrame(users)
df["num_messages"] = df["messages"].apply(len)
df["total_chars"] = df["messages"].apply(lambda msgs: sum(len(m) for m in msgs))
df["avg_msg_length"] = df["total_chars"] / df["num_messages"]

# Make severity columns ordered categoricals for correct plot ordering
df["grimt_sprog_severity"] = pd.Categorical(df["grimt_sprog_severity"], categories=SEVERITY_ORDER, ordered=True)
df["ludomani_severity"] = pd.Categorical(df["ludomani_severity"], categories=SEVERITY_ORDER, ordered=True)

df.head()

# %% [markdown]
# ## 2. Severity distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["grimt_sprog_severity"].value_counts().reindex(SEVERITY_ORDER).plot(
    kind="bar", ax=axes[0], color="#4C72B0"
)
axes[0].set_title("Grimt sprog severity distribution")
axes[0].set_xlabel("")
axes[0].set_ylabel("Antal brugere")

df["ludomani_severity"].value_counts().reindex(SEVERITY_ORDER).plot(
    kind="bar", ax=axes[1], color="#DD8452"
)
axes[1].set_title("Ludomani severity distribution")
axes[1].set_xlabel("")
axes[1].set_ylabel("Antal brugere")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "docs" / "severity_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Cross-tab: are the two categories independent in the data?
# Useful to confirm the dataset covers combinations like "Høj grimt sprog + Ingen ludomani"
# and not just correlated extremes.

# %%
crosstab = pd.crosstab(df["grimt_sprog_severity"], df["ludomani_severity"])
crosstab = crosstab.reindex(index=SEVERITY_ORDER, columns=SEVERITY_ORDER)

plt.figure(figsize=(6, 5))
sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Antal brugere"})
plt.title("Grimt sprog × Ludomani — kombinationer i datasættet")
plt.xlabel("Ludomani severity")
plt.ylabel("Grimt sprog severity")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "docs" / "severity_crosstab.png", dpi=150)
plt.show()

crosstab

# %% [markdown]
# ## 4. Message count & length stats

# %%
print(df[["num_messages", "avg_msg_length", "total_chars"]].describe())

# %%
plt.figure(figsize=(6, 4))
df["num_messages"].value_counts().sort_index().plot(kind="bar", color="#55A868")
plt.title("Antal beskeder pr. bruger")
plt.xlabel("Antal beskeder")
plt.ylabel("Antal brugere")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Example messages per severity level
# Pulled directly for use in the presentation.

# %%
def show_examples(category: str, n_per_level: int = 2):
    for level in SEVERITY_ORDER:
        subset = df[df[category] == level]
        print(f"\n--- {category} = {level} (n={len(subset)}) ---")
        for _, row in subset.head(n_per_level).iterrows():
            print(f"  user_id={row['user_id']}")
            for m in row["messages"][:2]:  # show first 2 messages only, keep it brief
                print(f"    - {m}")

print("=" * 70)
print("GRIMT SPROG EXAMPLES")
print("=" * 70)
show_examples("grimt_sprog_severity")

print("\n" + "=" * 70)
print("LUDOMANI EXAMPLES")
print("=" * 70)
show_examples("ludomani_severity")

# %% [markdown]
# ## 6. Edge cases & known label-quality issues
#
# During manual review of the generated dataset, two systematic patterns emerged
# in the local teacher model's (Mistral 7B-Instruct, Q4 quantized) output:
#
# 1. **Danish fluency**: occasional grammatical errors and invented words/phrasing,
#    consistent with known limitations of 7-8B instruction-tuned models on
#    lower-resource languages, likely compounded by the 8GB VRAM hardware
#    constraint capping model size and quantization level used.
#
# 2. **Grimt sprog label drift**: the model sometimes conflates *emotional intensity
#    about gambling losses* with *targeted hostility toward another user* — e.g.
#    labeling a message "Høj" for grimt sprog when the content expresses desperation
#    or admiration rather than insults directed at a specific person. This is a
#    task-following limitation distinct from the fluency issue, and would likely
#    persist even with a larger hosted model without tighter prompt constraints
#    or human-in-the-loop label verification.
#
# Below: examples flagged during manual review as likely mislabeled,
# kept in the dataset and documented rather than silently corrected,
# to preserve an honest record of data quality for evaluation purposes.

# %%
# Manually identified edge cases from review — user_ids found during spot-checking
edge_case_ids = [58, 63]  # example: update with the actual ids you flagged

edge_cases = df[df["user_id"].isin(edge_case_ids)]
for _, row in edge_cases.iterrows():
    print(f"\nuser_id={row['user_id']} | labeled grimt={row['grimt_sprog_severity']}, ludomani={row['ludomani_severity']}")
    for m in row["messages"]:
        print(f"  - {m}")
    print("  Note: review for label accuracy — see markdown above for known issue pattern")

# %% [markdown]
# ## Summary
# - 150 synthetic users, even split (~25%) across 4 severity levels for both categories
# - All 16 grimt×ludomani combinations represented (min cell count: see crosstab above)
# - Known limitations documented above; dataset used as-is for the training step,
#   given the scope and time constraints of this case exercise