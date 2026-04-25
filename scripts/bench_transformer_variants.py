#!/usr/bin/env python3
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent.parent / "experiments" / "step1_engram_core" / "scripts" / "bench_transformer_variants.py"), run_name="__main__")
