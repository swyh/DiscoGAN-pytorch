# DiscoGAN-pytorch
I brought the model, loss code of the official DiscoGAN.

Official paper : [https://arxiv.org/pdf/1703.05192.pdf](https://arxiv.org/pdf/1703.05192.pdf) <br>
Official implement : [https://github.com/SKTBrain/DiscoGAN](https://github.com/SKTBrain/DiscoGAN)

## Prerequisites
- Python 3.6
- PyTorch
- Numpy/Scipy/Pandas
- Progressbar
- OpenCV

## Execution

### Dataset Download
see Official DiscoGAN github

### Training
- dataset directory
  ```
  - DiscoGAN-pytorch
   - train
    - A
    - B
  ```
- train

       $ python train.py
    
    
- Road Model<br>
    
        $ python train.py --load_epoch -4 --model_path ./models       
   you should set to load epoch, model_path (defualt epoch : -0, model_path : ./models)

 
### Test
- dataset directory
  ```
  - DiscoGAN-pytorch
   - test
    - A
    - B
  ```
- test


      $ python test.py
    
    
