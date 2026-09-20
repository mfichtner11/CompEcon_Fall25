#!/bin/bash

echo "Creating results directory..."
#####
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
#####


