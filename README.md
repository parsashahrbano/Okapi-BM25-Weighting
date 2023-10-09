


This repository is an implementation of "Okapi BM25 Weighting in Graph-Based Extractive Multi-Document Summarization". After downloading the code, you need to import two types of documents: one for relevant document(s) and one for non-relevant document(s). This is necessary because the calculation of the topic signature requires non-relevant data as well. You can run the program from the terminal with the following command:

```markdown
python harmonic-main-BM25.py "relevant_path" "nonrelevant_path"
```

If you want to use a bash script to run all documents and clusters (like DUC2004, which contains 50 clusters) one by one, you can run the following file: "run-bash.sh". You will need to replace all destinations in the script. To run the script, use:

```markdown
./run-bash.sh
```
