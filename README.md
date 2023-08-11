# CBS
Official Code of [The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another in Neural Networks](https://proceedings.mlr.press/v162/yu22f.html)[ICML2022]

1. Setup the enviroment:
    ```
    conda create -n cbs python=3.6.13
    conda activate cbs
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    sh setup.sh
    
    ```
2. Prepare data
    ```
    mkdir prob_regressor_data
    mkdir prob_regressor_results
    mkdir checkpoints
    ```
    Please download checkpoints from [here](https://drive.google.com/drive/folders/18ix239cy261ug_IGZbhtYKPzkkniTyee?usp=sharing) and put them in the "checkpoints" folder. 

3. Run Entropic Wasserstein Pruning:
   3.1 on MLPNet
   ```
       sh scripts/sweep_mnist_mlpnet_ot.sh
   ```
   3.2 on ResNet20
   ```
       sh scripts/sweep_cifar10_resnet20_ot.sh
   ```
   3.3 on MobileNetV1
   ```
       sh scripts/sweep_imagenet_mobilenet_ot.sh
   ```
   

## Citaton
We thank Singh & Alistarh for sharing their code of [WoodFisher](https://github.com/IST-DASLab/WoodFisher). We thank also Yu, Xin and Serra et. al. for the CBS code [CBS](https://github.com/yuxwind/cbs), from which the repository is forked.

Our implementation is based on their code. If our work is helpful to your research/project, please cite our work.
