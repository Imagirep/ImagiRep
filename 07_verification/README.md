# Verification

`scripts/verify_from_raw.py` recomputes every number reported in the paper (tables, prose and benchmark statistics)
from the stored per-question outputs and compares it with the paper value. It makes no model calls.

```bash
cd 07_verification/scripts && python verify_from_raw.py
```

It needs the `benchmark/` folder from the Hugging Face dataset. The expected output is in `verification_output.txt`
(424 checks, 0 mismatches); `verification_values.json` holds the same values. Each check names the files it reads, so
the script also serves as the map from every number to its sources.
