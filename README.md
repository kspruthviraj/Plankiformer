# Plankiformer

# Zoofier
Code for plankton dataset creation and classification based on Keras.

---

## Quick Summary

Use `analyze_dataset.py` to analyze the datasets.

Use `train.py` to train models on plankton data.

Use `train_from_saved.py`, to train models on saved data. For instance, you can use this command to reproduce our results using the data provided with the same train,test and valid split.

Use `predict.py`, to classify images using a single model.

Use `predict_ensemble.py`, to classify images that does not have labels using ensemble of models. 

---

## Repo Structure

The repo contains the following directories:

- `utils`: contains auxiliary code.

- `out`: the output is stored here (no output is uploaded to GitHub, so it must be created).

- `Data`: the input data is stored here.

- `trained-models`: contains only best trained models for the end users.


## Training models

In order to train a fresh model, use `train.py`. 

```python
python main.py
```
There are lots of input commands that can be given to the script. To query them, use the `-h` flag (`python train.py -h`). 
