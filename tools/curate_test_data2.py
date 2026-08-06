"""This script is designed to isolate the 26 human performance tracking lines for Beethoven's WoO 80,
ensuring that only the relevant entries are retained for further analysis.
"""

import hashlib
import shutil
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def isolate_beethoven_woo80_records(metadata_csv: Path) -> pd.DataFrame:
    """Isolates the 26 human performance tracking lines for Beethoven's WoO 80."""
    df = pd.read_csv(metadata_csv)

    # 1. Filter by Composer first to secure the namespace
    beethoven_df = df[df["canonical_composer"].str.strip() == "Ludwig van Beethoven"]

    # 2. Case-insensitive substring query targeting the unique catalog number 'woo 80'

    # This captures both 'WoO 80' and 'WoO80' variants automatically
    woo80_tracks = beethoven_df[
        beethoven_df["canonical_title"].str.contains(
            r"WoO\s*80", na=False, case=False, regex=True
        )
    ]

    # Optional verification printout to double-check the visual feedback names
    print(f"🎯 Successfully isolated {len(woo80_tracks)} true records for WoO 80.")
    if not woo80_tracks.empty:
        print(woo80_tracks.iloc[0:5, [0, 1, 5]])  # Safe position-based fallback lookup

    return woo80_tracks


def download_isolated_woo80_assets(
    filtered_tracks: pd.DataFrame, output_dir: Path
) -> None:
    """Downloads isolated Beethoven WoO 80 assets safely using a robust mirror stream."""

    audio_out = output_dir / "audio"
    midi_out = output_dir / "midi"
    audio_out.mkdir(parents=True, exist_ok=True)
    midi_out.mkdir(parents=True, exist_ok=True)

    # Stream the dataset mirror configuration to download files on-demand
    print("✨ Initializing safe streaming gateway for MAESTRO v3.0.0 data...")
    ds_stream = load_dataset(
        "projectlosangeles/maestro-v3.0.0", split="train", streaming=True
    )

    # Map your target files to a quick verification set
    target_filenames = set(
        filtered_tracks["audio_filename"].apply(lambda x: Path(x).name)
    )
    download_count = 0

    print(
        f"⏳ Scanning archive stream for {len(target_filenames)} specific WoO 80 files..."
    )
    for item in ds_stream:
        # Check if the streamed file matches our isolated target array
        audio_meta = item["audio"]
        current_name = Path(audio_meta["path"]).name

        if current_name in target_filenames:
            download_count += 1

            perf_id = hashlib.md5(current_name.encode("utf-8")).hexdigest()[:12]

            local_wav = audio_out / f"{perf_id}.wav"
            local_mid = midi_out / f"{perf_id}.mid"

            if not local_wav.exists():
                print(
                    f"   📥 [{download_count}/{len(target_filenames)}] Streaming Audio: {perf_id}.wav"
                )
                shutil.copy(audio_meta["path"], local_wav)

            # If you also have a matching midi track streamed, copy it similarly
            if "midi" in item and not local_mid.exists():
                shutil.copy(item["midi"]["path"], local_mid)

            if download_count >= len(target_filenames):
                break

    print("\n✅ Beethoven evaluation testing dataset compiled successfully!")


if __name__ == "__main__":
    # Point this to your local catalog download path location
    isolated_df = isolate_beethoven_woo80_records(Path("ref/maestro-v3.0.0.csv"))
    download_isolated_woo80_assets(
        filtered_tracks=isolated_df,
        output_dir=Path("../data/assets/recordings/Beethoven_WoO80"),
    )
