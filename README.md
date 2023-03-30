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


# Example Performance reports on the Phytoplankton and Zooplankton images using EfficientNet-B7 networks.


 Accuracy

0.9025121859767529

F1 Score

0.8838444691595628

Classification Report

                                  precision    recall  f1-score   support

                      ascomorpha       1.00      1.00      1.00        13
                       askenasia       0.98      0.98      0.98        59
                       asplachna       0.00      0.00      0.00         2
           asterionella_colonies       1.00      0.94      0.97        68
            asterionella_partial       0.60      0.75      0.67        12
                     aulacoseira       0.85      0.87      0.86        39
                 centric_diatoms       0.99      0.99      0.99       158
                        ceratium       0.97      0.98      0.98       179
                     chlorophyte       0.85      0.78      0.81       123
   chlorophyte_colonial_dividing       0.92      0.85      0.89       172
           chlorophyte_elongated       0.85      0.87      0.86        71
               chlorophyte_frame       0.93      1.00      0.96        39
              chlorophyte_square       0.90      0.94      0.92        65
                   chroococcales       0.87      0.96      0.91        27
                   ciliate_round       0.72      0.84      0.78        25
                        ciliates       1.00      0.88      0.94        17
                   ciliates_blue       0.93      0.96      0.95        85
                  ciliates_green       0.93      0.95      0.94        60
                      closterium       0.98      0.98      0.98       128
          coelastrum_reticulatum       0.93      0.96      0.95        28
                  coelosphaerium       0.85      0.90      0.87        68
                          coleps       1.00      1.00      1.00        10
                       cosmarium       0.99      0.97      0.98        87
       cryptomonas_cryptophyceae       0.94      0.94      0.94       241
             cryptophytes_blurry       0.72      0.91      0.81        68
     cyanobacteria_colonial_blue       0.90      0.88      0.89        96
cyanobacteria_colonial_clathrate       0.78      0.87      0.82        45
 cyanobacteria_colonial_probably       0.94      0.92      0.93       167
       cyanobacteria_filamentous       0.95      0.89      0.92        83
                        didinium       0.69      0.90      0.78        10
                       dinobryon       0.93      0.96      0.95       170
           dinobryon_single_cell       0.88      0.87      0.87       105
          dinoflagellate_diamond       0.98      0.96      0.97        57
                  dolichospermum       0.90      0.99      0.94        93
                    elakatothrix       0.91      0.98      0.94        59
                        filament       0.94      0.92      0.93       100
                      fragilaria       0.98      0.98      0.98       170
                  gomphosphaeria       1.00      1.00      1.00         2
                          gonium       0.88      1.00      0.94        15
                     gymnodinium       0.93      0.89      0.91        46
                  hormidium_like       0.94      0.99      0.97       117
                     kellicottia       1.00      1.00      1.00         6
            keratella_cochlearis       0.89      0.89      0.89        55
              keratella_quadrata       0.50      0.33      0.40         6
                     limnoraphis       0.77      0.91      0.83        11
            mallomonas_akrokomos       0.62      0.83      0.71        18
                  mallomonas_big       0.88      0.98      0.93        65
                         nauplii       0.94      0.94      0.94        18
                     oocystaceae       0.87      0.95      0.91        80
                       pandorina       0.93      1.00      0.96        50
                    paradileptus       1.00      0.90      0.95        10
                      pediastrum       1.00      1.00      1.00        42
                  pennate_diatom       0.85      0.71      0.77        48
                      peridinium       0.91      0.92      0.92        92
                        phacotus       0.80      0.97      0.88        37
                   plankton_halo       0.93      0.93      0.93        14
                      plankton_y       1.00      1.00      1.00         7
                      plankton_z       0.89      1.00      0.94        24
                 planktosphaeria       0.80      1.00      0.89         4
                    planktothrix       0.91      0.91      0.91        23
                      polyarthra       0.96      0.88      0.92        56
            protist_like_ciliate       0.86      0.94      0.90       129
                      rhodomonas       0.98      0.90      0.93        88
                         rotifer       0.52      0.67      0.58        24
                    rotifer_long       0.75      0.60      0.67         5
                       rotifer_z       0.93      1.00      0.96        13
                     scenedesmus       0.92      0.94      0.93        50
                     staurastrum       0.98      1.00      0.99       110
                     strombidium       0.96      0.93      0.95        87
                       synchaeta       0.93      0.93      0.93        14
                         synedra       0.85      0.92      0.88        50
            synedra_angustissima       0.92      0.92      0.92        13
                          synura       0.93      1.00      0.96        13
                      tetraedron       1.00      1.00      1.00        26
                     tintinidium       0.84      0.94      0.89        17
                     tintinopsis       0.64      1.00      0.78         7
                     trichocerca       0.91      0.94      0.93        33
                         unknown       0.68      0.50      0.58       232
               unknown_eccentric       0.67      0.66      0.66        87
               unknown_elongated       0.82      0.61      0.70        97
           unknown_probably_dirt       0.95      0.92      0.93       114
             unrecognizable_dots       0.99      0.98      0.99       152
                        uroglena       0.93      1.00      0.96        13
       vorticella_epistylis_like       0.86      0.88      0.87        43
                     zooplankton       0.82      0.92      0.87        72

                        accuracy                           0.90      5334
                       macro avg       0.87      0.90      0.88      5334
                    weighted avg       0.90      0.90      0.90      5334




