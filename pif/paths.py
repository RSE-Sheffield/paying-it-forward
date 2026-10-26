"""Where everything lives. The five factor registries sit together under factors/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FACTORS = ROOT / "factors"
DATASETS = FACTORS / "datasets"
CRITERIA = FACTORS / "criteria"
PROMPTS = FACTORS / "prompts"
INPUTS = FACTORS / "inputs"
REVIEWERS = FACTORS / "reviewers"
EXPERIMENTS = ROOT / "experiments"
# Trial runs (--limit) go here so they never mix with real results.
SCRATCH = EXPERIMENTS / "_scratch"
SECRETS = ROOT / ".secrets"
