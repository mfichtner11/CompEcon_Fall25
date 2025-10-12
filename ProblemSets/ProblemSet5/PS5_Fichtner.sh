#!/bin/bash

echo "Starting Problem Set 5..."

echo "Step 1: Data Import"
Rscript ImportandCombine.R

echo "Step 2: Variables"  
python3 CreateVariables.py

echo "Step 3: Cleaning"
python3 DataCleaning_PS5_Fichtner.py

echo "Step 4: Visualization"
python3 Visualization_PS5_Fichtner.py

echo "Step 5: Regressions"
python3 Regressions.py

echo "Step 6: LaTeX"
pdflatex ProblemSet5Fichtner.tex

echo "Done!"