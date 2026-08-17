from pathlib import Path


ROOTS = [
    Path("data/raw"),
    Path("data/external/processed"),
]


MIN_WORDS = 5


def validate_file(path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return False, "empty"

    words = text.split()

    if len(words) < MIN_WORDS:
        return False, "too_short"

    return True, "ok"


checked = 0
passed = 0
failed = 0


for root in ROOTS:

    if not root.exists():
        continue

    for path in sorted(root.rglob("*.txt")):

        checked += 1

        ok, reason = validate_file(path)

        if ok:
            passed += 1
            print("PASS:", path)
        else:
            failed += 1
            print("SKIP:", path, "|", reason)


print()
print("LUMIVEX DATASET VALIDATION COMPLETE")
print("Checked:", checked)
print("Passed:", passed)
print("Skipped:", failed)
