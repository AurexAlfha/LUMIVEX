from pathlib import Path
import csv
import shutil


VERIFIED_DIR = Path("data/external/verified")
PROCESSED_DIR = Path("data/external/processed")
REGISTRY = Path("data/external/dataset_registry.csv")


def load_registry():
    approved = set()

    if not REGISTRY.exists():
        return approved

    with REGISTRY.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        for row in csv.DictReader(f):

            status = row.get("status", "").strip().upper()
            training = row.get("training_status", "").strip().upper()

            if (
                status in {"VERIFIED", "PUBLIC DOMAIN"}
                and training == "APPROVED"
            ):
                approved.add(row["work"].strip())

    return approved


def import_verified_text():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    approved = load_registry()

    imported = 0

    for source in sorted(VERIFIED_DIR.glob("*.txt")):

        if source.name == "gutenberg_sample.txt":
            continue

        if source.stem not in approved:
            print(
                "SKIPPED (not registry-approved):",
                source.name
            )
            continue

        shutil.copy2(
            source,
            PROCESSED_DIR / source.name
        )

        imported += 1

    print("LUMIVEX VERIFIED IMPORT COMPLETE")
    print("Approved imported files:", imported)


if __name__ == "__main__":
    import_verified_text()
