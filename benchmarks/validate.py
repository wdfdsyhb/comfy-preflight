"""Validate benchmarks/data.json entries.

Run from anywhere: python benchmarks/validate.py
Prints problems to stdout; exits 1 if any entry is invalid.
"""
import json
import os
import re
import sys

FAMILIES = {"sd15", "sdxl", "sd35m", "sd35l", "flux", "qwen"}
MODES = {"normal", "lowvram", "medvram", "novram"}
REQUIRED = ("gpu", "vram_total_gb", "model", "family", "resolution",
            "batch", "mode", "peak_vram_gb", "measured_with", "date")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    errors = []
    keys = set()
    for i, e in enumerate(entries):
        tag = "entry[%d]" % i
        for field in REQUIRED:
            if field not in e:
                errors.append("%s: missing field '%s'" % (tag, field))
        if e.get("family") not in FAMILIES:
            errors.append("%s: family '%s' not in %s" % (tag, e.get("family"), sorted(FAMILIES)))
        if e.get("mode") not in MODES:
            errors.append("%s: mode '%s' not in %s" % (tag, e.get("mode"), sorted(MODES)))
        res = e.get("resolution")
        if not (isinstance(res, list) and len(res) == 2
                and all(isinstance(x, int) and x > 0 for x in res)):
            errors.append("%s: resolution must be [width, height] ints" % tag)
        for num in ("vram_total_gb", "peak_vram_gb"):
            v = e.get(num)
            if not (isinstance(v, (int, float)) and v > 0):
                errors.append("%s: %s must be a positive number" % (tag, num))
        if isinstance(e.get("peak_vram_gb"), (int, float)) and isinstance(e.get("vram_total_gb"), (int, float)):
            if e["peak_vram_gb"] > e["vram_total_gb"] * 1.15:
                errors.append("%s: peak_vram_gb exceeds vram_total by >15%% (typo?)" % tag)
        if not DATE_RE.match(str(e.get("date", ""))):
            errors.append("%s: date must be YYYY-MM-DD" % tag)
        key = (e.get("gpu"), e.get("model"), tuple(res or []), e.get("batch"), e.get("mode"))
        if key in keys:
            errors.append("%s: duplicate entry (same gpu/model/res/batch/mode)" % tag)
        keys.add(key)
    if errors:
        print("INVALID (%d problems):" % len(errors))
        for err in errors:
            print("  - %s" % err)
        return 1
    print("OK: %d entries valid." % len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
