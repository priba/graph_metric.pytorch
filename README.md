# Graph Metric Learning

Graph Metric Learning in PyTorch.

## Install

- Install all the requirements. 
    $ conda env create -f environment.yml


## Usage

    $ conda activate graphmetric

### Train

* Write configuration file. Follow the example [here](./config/).
* Run the training script with the corresponding configuration file `./train.sh config/train.cfg`

### Test

* Write configuration file. Follow the example [here](./config/) providing a load path (`--load`).
* Run the test script with the corresponding configuration file `./test.sh config/test.cfg`

### Tensorboard

* Setup your server running the script with the corresponding path `./board.sh ./checkpoints/`
* Go to your [tensorboard page](localhost:6006) using your browser.

## Author

* [Pau Riba](http://www.cvc.uab.es/people/priba/) ([@priba](https://github.com/priba))

