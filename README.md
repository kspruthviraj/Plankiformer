# Plankiformer

## Repo Structure

The repo contains the following directories:

- `utils`: contains auxiliary code.

- `out`: the output is stored here (no output is uploaded to GitHub, so it must be created).

- `Data`: the input data is stored here.

- `trained-models`: contains only best trained models for the end users.


## Training models

In order to train a fresh model, use `main.py`. 

The training depends on how the data is structured.
There are basically three scenarios:
1) You have directory without the train, test and valid split
2) You have directory already with train and test split
3) The data is in cloud and can be downloaded. Ex: Cifar-10, Stanford-dogs data etc.

For the first case:
You can train the model using:

```python
python main.py -datapaths ./data/PhytoData/ -outpath ./out/phyto_out/ -classifier multi -aug -datakind image -ttkind image -save_data yes -resize_images 1 -L 128 -valid_set yes -test_set yes -dataset_name zoolake -training_data False -epochs 40 -finetune 2 -finetune_epochs 40 -balance_weight yes -batch_size 32 -init_name Init_0
```
There are lots of input commands that can be given to the script. To query them, use the `-h` flag (`python main.py -h`). 
