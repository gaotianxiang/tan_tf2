# TAN in Tensorflow 2.0

## Structure of the repo
- All the transformations are defined in `./module/transforms.py`
- Two different loss functions (prior) are defined in `./module/conditional.py`
- All the dataloaders are defined in `./dataloader`
- `trainer.py` includes functions for training and testing.
- `main.py` has the main function to run the program.

## Run the codes
To run the code, simpy run the command below.

```
python main.py
```
This will give results of a RealNVP model trained on BSDS300 dataset. 

We can define any flow-based models using `transforms.Transformer()` which takes a list of transformers as input. 

To change dataset, modify the following codes in `main.py`
```python
# get data loaders
dtst = dataloader.BSDS300()
train_dl, valid_dl, test_dl = dtst.get_dl(batch_size)
```

Valid options:
- `dtst = dataloader.GAS()`
- `dtst = dataloader.Power()`
- `dtst = dataloader.Hepmass()`
- `dtst = dataloader.Miniboone()`
- `dtst = dataloader.BSDS300()`


## Requirements
- python 3.7
- Tensorflow 2.1


