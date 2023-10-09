#!/bin/bash

# Set environment variables
DUC="/home/shishi/penv/DUCs/DUC2003clean/"
corpus="/home/shishi/penv/DUCs/DUC2003clean/corpus/"
main="/home/shishi/penv/DUCs/DUC2003clean/main"
store="/home/shishi/penv/DUCs/result2003-arithmetic-harmonic/"
code="/home/shishi/penv/weightedschemes_chameleon"
# Change to the DUC directory
cd "$DUC"

# Loop over files in the corpus directory
for anyfile in "$corpus"/*
do
    # Move the file to the main directory
    mv "$anyfile" "$main"

    # Get the filename without the directory path
    filename=$(basename "$anyfile")

    # Check if the file already exists in the store directory
    if [ -e "$store/$filename" ]; then
        # Run the Python script and append the output to the harmonic.txt file
        python $code/harmonic-main-BM25.py "$main/$filename" "$corpus" > "$store/$filename/5_ws_frac_5_2003.txt"
    fi

    # Move the file back to the corpus directory
    mv "$main/$filename" "$corpus"
done
