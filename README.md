# CBSA

## Requirements：

* python=3.8 
* torch=1.8.0
* sklearn=1.2.2

## Usage

1. Pre-train the model with
```python
python run.py --model_name bert-base-multilingual-cased --dataset PubMed --pretrained_dataset PubMed --train True
```
2. Fine-tune the model with 
```python
python run.py --model_name bert-base-multilingual-cased --dataset PubMed --pretrained_dataset CBSA --train False
```