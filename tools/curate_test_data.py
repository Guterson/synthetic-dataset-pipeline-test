"""A diagnostic utility to analyze metadata and extract compositions
with the highest performance density under text instability constraints.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from thefuzz import fuzz


def find_highest_density_compositions(
    csv_path: Path, top_n: int = 10, similarity_threshold: int = 85
) -> None:
    """Groups unstable string titles via fuzzy token ratios
    to locate pieces with maximum real renditions."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Target catalog missing: {csv_path}")

    # Read the master catalog sheet
    df: DataFrame = pd.read_csv(csv_path)

    # 1. Isolate the dataset down to clean tracking series primitives
    df["canonical_composer"] = df["canonical_composer"].str.strip()
    df["canonical_title"] = df["canonical_title"].str.strip()

    # Group by composer first to drastically minimize our fuzzy calculation matrix space
    composers: np.ndarray = df["canonical_composer"].unique()
    master_vetted_groups: list[dict] = []

    for composer in composers:
        composer_df = df[df["canonical_composer"] == composer].copy()
        unique_titles = composer_df["canonical_title"].unique()

        # Track which titles have already been merged into a canonical identity
        vetted_titles: dict[str, list[str]] = {}

        for title in unique_titles:
            matched_canonical = None

            # Check if this title is a fuzzy variation of a title we already cataloged
            for canonical_name in vetted_titles:
                # Token Set Ratio handles variations like inverted words or extra spaces flawlessly
                score = fuzz.token_set_ratio(title.lower(), canonical_name.lower())
                if score >= similarity_threshold:
                    matched_canonical = canonical_name
                    break

            if matched_canonical:
                vetted_titles[matched_canonical].append(title)
            else:
                vetted_titles[title] = [title]

        # 2. Re-map the raw df rows to our consolidated fuzzy keys and count them
        for canonical, variants in vetted_titles.items():
            total_performances = composer_df[
                composer_df["canonical_title"].isin(variants)
            ]
            count = len(total_performances)

            master_vetted_groups.append(
                {
                    "composer": composer,
                    "canonical_title": canonical,
                    "variant_count_detected": len(variants),
                    "total_renditions": count,
                    "example_variants": list(set(variants))[:2],
                }
            )

    # 3. Sort, clean, and project the final rankings to the screen
    rankings_df = pd.DataFrame(master_vetted_groups)
    top_pieces = rankings_df.sort_values(by="total_renditions", ascending=False).head(
        top_n
    )

    print("\n📊 === TOP MAESTRO PIECES BY TOTAL HUMAN RENDITIONS ===")
    for idx, (_, row) in enumerate(top_pieces.iterrows(), start=1):
        print(f"{idx}. {row['composer']} - {row['canonical_title']}")
        print(
            f"   🔹 Total Recorded Renditions: {row['total_renditions']} (Across {row['variant_count_detected']} text layout variations)"
        )
        print(f"   🔹 Naming Examples found: {row['example_variants']}\n")


if __name__ == "__main__":
    # Point this to your local catalog download path location
    find_highest_density_compositions(Path("ref/maestro-v3.0.0.csv"))
