# Viterbi Algorithm Implementation for Tagging

## Performance on ner_dev.dat 
Found 5832 NEs. Expected 5931 NEs; Correct: 4241.
F1-Score 0.721075

## Train and Predict
```Python
python viterbi.py
```
## Evaluate
```Python
python eval_ne_tagger3.py ner_dev.key predict_result.txt
```
