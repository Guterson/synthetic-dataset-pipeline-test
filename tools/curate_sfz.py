"""Terminal task wrapper to scan and optimize all SFZ files in the asset directory."""

from pathlib import Path

from score2dataset.processors.sfz_conditioner import SfzConditioner


def main() -> None:
    """Discovers all local SFZ definitions and applies onset transient conditioning."""
    project_root: Path = Path(Path(__file__).resolve().parents[1])
    sfz_asset_root: Path = project_root / "data" / "assets" / "sfz"

    if not sfz_asset_root.exists():
        print(f"[ERROR] Asset directory not found: {sfz_asset_root}")
        return

    # Dynamic glob discovery across all subdirectories
    sfz_files: list[Path] = list(sfz_asset_root.rglob("*.sfz"))
    print(f"Found {len(sfz_files)} file(s) inside the SFZ library pool.")

    for sfz_path in sfz_files:
        # Prevent the tool from reprocessing files it has already fixed on previous runs
        if sfz_path.stem.endswith("_OnsetSafe"):
            continue

        target_output_path = sfz_path.parent / f"{sfz_path.stem}_OnsetSafe.sfz"

        print(f"\nProcessing target configuration: {sfz_path.name}")
        conditioner = SfzConditioner(sfz_path=sfz_path)
        conditioner.process_and_fix(output_path=target_output_path)


if __name__ == "__main__":
    main()
