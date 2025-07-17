# A source-free unsupervised domain adaptation method for cross-regional and cross-time crop mapping from satellite image time series
The official implementation code for our paper "A source-free unsupervised domain adaptation method for cross-regional and cross-time crop mapping from satellite image time series".


Our paper has been accepted to Remote Sensing of Environment and is publicly available at: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0034425724004115).


### Requirements
- [numpy 1.23.5](https://numpy.org/)
- [torch 1.12.1+cu116](https://pytorch.org/)
- [scikit_learn 1.1.1](https://scikit-learn.org/)

### Usage

1- Download the preprocessed data for the three sites in the USA from [Google Drive](https://drive.google.com/file/d/1T_gZFC_njzXU7vG3SawZ92gH6ms9ffT5/view?usp=sharing) - Then extract the ZIP file into the root directory.

2- Pretrain the model on the source domain data using the following code:

```
python source_training.py --pretrained_save_dir Pretrained_USA --backbone_network CNN --source_site A --source_year 2019 --data_dir Data_USA
```

- *backbone_network*: You can select either 'CNN', 'Transformer', or 'LSTM'.

- *source_site*: You can select either 'A','B',or 'C'. Please note that Sites A, B, and C correspond to Sites IA, MO, and MS in the paper, respectively.

- *source_year*: You can select either '2019','2020',or '2021'.

- *pretrained_save_dir*: Set the path for saving the pretrained model.

- *data_dir*: Path where the data of the three sites are located.


3- Apply domain adaptation to the target domain data and evaluate the performance using the following code:

```
python AdaptationandEvaluation.py --adapted_save_dir Adapted --pretrained_save_dir Pretrained_USA --backbone_network CNN --source_site A --target_site C --source_year 2019 --target_year 2021
```

- *target_site*: You can select either 'A','B',or 'C'.

- *target_year*: You can select either '2019','2020',or '2021'.

- *adapted_save_dir*: Set the path for saving the adapted models.



## Citation
```
@article{mohammadi2024source,
  title={A source-free unsupervised domain adaptation method for cross-regional and cross-time crop mapping from satellite image time series},
  author={Mohammadi, Sina and Belgiu, Mariana and Stein, Alfred},
  journal={Remote Sensing of Environment},
  volume={314},
  pages={114385},
  year={2024},
  publisher={Elsevier}
}
```
