# Computer Vision for Marine Life Classification

## Table of Contents

1. [Objectives](#objectives)
2. [Running](#running)
3. [Introduction](#introduction)
4. [Evaluation](#evaluation)
5. [Dataset](#dataset)
6. [Challenges](#challenges)

## Objectives

To explore the capabilities of differing machine learning architecture for the classification of various species of marine biology.

## Running

Run `make venv` to setup the virtual environment required to run the notebook.

Notes:

_Access to the S3 bucket is required to download the dataset, alternative methods may be provided in the future. The dataset on the S3 bucket is a subset of the [iNat Challenge 2021 - FGVC8](https://www.kaggle.com/c/inaturalist-2021) dataset._

_Running on a CUDA capable system will accelerate training and prediction by many magnitudes compared to just a CPU._

## Introduction

With there being thousands upon thousands of species of marine life, it takes years of expertise to be able to identify the species from a single image. For many, devoting years just to identify species is not feasible, and although there are existing machine learning models to classify marine life, they are generally closed-source and/or locked behind a paywall. We look to tackle this complex multi-class image classification problem while keeping our methods free and open-source. Leveraging various machine learning architectures such as transformers and convolutional neural networks, the goal is to be able to accurately classify the marine species in the image and compare the results between the different approaches.

Multiple groups of researchers have explored the potential of convolutional neural networks for marine life classification, yet their projects differed from ours quite significantly. One group decided to classify only 9 species of marine life, with around 1,100 images for each species being used for training, making their scope vastly smaller than our project \cite{Dey_2023}. Another group had a larger scope than ours and covered more species, but used generalized classifications that encapsulated multiple species into one \cite{10441585}. We look to take a different approach in that we want to investigate the limits of exactly how granular we can make an image classification model. In addition, our incorporation of a visual transformer model is not a direction that was explored by these other groups of researchers, and we are curious of how well this architecture can perform compared to convolutional neural networks.

## Evaluation

We plan to take two main approaches to our image classification task. The first is a convolutional neural network, and the second is a visual transformer. Both will be evaluated using the same accuracy metrics: top-1 error rate and top-5 error rate (a lower number is better), in order to reflect the format of the Kaggle competition associated with our dataset. Specifically, these error rates will take the form of simple decimals, describing the proportion of classifications that were incorrect out of all classifications made by our models. At the same time, we will also investigate the precision of each available species to see whether certain fish are predicted more precisely than others. While recall will be calculated as well, we prefer accuracy and precision as our main metrics because there is little consequence to incorrectly classifying species.

## Dataset

The dataset that is being used is sourced from the [iNat Challenge 2021 - FGVC8](https://www.kaggle.com/c/inaturalist-2021) Kaggle competition. The data are user submitted images from the platform iNaturalist, which is a community where naturalists can share their findings to both enthusiasts and the scientific community. In the whole dataset, there are over 2.7 million images (amounting to almost 300GB) and 10,000 species. However, to narrow the scope, only marine species will be used, which reduces the species count to less than 200. This cuts down the data to only around 10GB. Each species has 200-300 images of training images and 10 validation images, where each image has a max dimension of 500 pixels.

This Kaggle dataset also has a pre-determined split of training and validation data, which we will use. Once trained, we will verify our results using the validation dataset, and then calculate our error rates from those classifications. We would have liked to use the testing split of the Kaggle dataset as well, but due to the nature of a modeling competition, that data is not released publicly and therefore we can only use the training and validation sets.

The taxonomic rank of class was used to filter the species down to marine life only, leaving us this exhaustive list:

- **Actinopterygii** (Ray-finned fishes)
  - Tuna, swordfish, seahorses
- **Gastropoda** (Slugs and snails)
- **Malacostraca** (Crustaceans)
  - crabs, lobsters, krill
- **Bivalvia** (Bivalves)
  - Mussels, clams, scallops
- **Anthozoa** (Sea anemones and corals)
- **Elasmobranchii** (Cartilaginous fish)
  - Sharks, rays, skates, sawfish
- **Asteroidea** (Starfish or sea stars)
- **Polyplacophora** (Chitons)
- **Hexanauplia**
  - Zooplankton and barnacles
- **Echinoidea**
  - Sea urchins or urchins
- **Scyphozoa** (True jellyfish)
  - Lion's mane jellyfish, Moon jellyfish
- **Cephalopoda**
  - Squid, octopus, cuttlefish, nautilus
- **Hydrozoa** (Hydrozoans)
  - Freshwater jelly, Portuguese man o' war
- **Ascidiacea** (Ascidians or sea squirts)
- **Holothuroidea** (Sea cucumbers)
- **Ophiuroidea**
  - Brittle stars, serpent stars, or ophiuroids

However, for the purposes of this project, the classes of Actinopterygii and Elasmobranchii were selected to represent the vast majority of fish with 183 and 16 species respectively, totaling to 199 species rather than 478 species if all marine classes were used. We believe that narrowing down the scope of species that we are classifying to only fish will not only increase the likelihood of achieving accurate results, but will also enable faster iterations when re-training and fine-tuning our models.

## Challenges

Binary image classification is already a computationally expensive task already, yet we are upping the complexity by introducing a much larger domain of classes that are being trained for and predicted from. We are only using a subset of the data already, but there are still almost 200 species to deal with. However, we will be able to leverage the Northeastern University Discovery Cluster, which is a high-performance computer resource used by the Northeastern University research community. Because of this, we should be able to iterate the training and evaluation much faster than if we were to use our personal computers.

Another difficulty is the quality of the data. Since the images are user submitted, they may be incorrectly classified or the image quality may be sub optimal. Some images are taken out of their natural habitat, while others are ideal. There low quality data is a small percentage of our dataset, but may need to be manually reviewed for quality as generally the quality of the model depends on the quality of the input.
