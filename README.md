# Stealthy Query-Efficient Black-Box Attack against Interpretable Deep Learning

Deep neural network (DNN) models are susceptible to adversarial samples in white- and black-box environments.
Although previous studies have shown high attack success rates, coupling DNN models with interpretation models could offer a sense of security when a human expert is involved, who can identify whether a given sample is benign or malicious. However, in white-box environments, interpretable deep learning systems (IDLSes) have been shown to be vulnerable to malicious manipulations.

## Install environment

Use `my_env.yaml` file to install correct versions of packages and to run the code

## How to run the code

Without defense:

The code generates adversarial samples based on the ResNet-50 model. Target model for the following script is set to InceptionV3.

```sh
python main.py --ensemble_models resnet50 --model inceptionv3 --dataset imagenet --epsilon 0.047 --max_queries 10000 --mr 0.001 --num_attack 1000
```

With defense:

```sh
python main.py --ensemble_models resnet50 --model inceptionv3 --dataset imagenet --epsilon 0.047 --max_queries 10000 --mr 0.001 --num_attack 1000 --defense_method RP
```

The examples are provided in the *example* folder (one image per category). The results will be saved in the *output* folder. 

The table below describes each field

Field  | Field Description
------------- | -------------
ensemble_models  | The source model to generate adversarial images, default is ResNet50
model  | Target black-box model to be attacked 
dataset | source images to generate adversarial samples
epsilon | Threshold to add perturbation
max_queries | Maximum number of queries to send to the blackbox model
mr | Mutation rate for the genetic algorithm
num_attack | the amount of images to be attacked
defense_method | Defense algorithm to test the attack robustness (*RP, BitDepthReduce, JPEG, MedianSmoothing2D*)
