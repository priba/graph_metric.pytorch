# Graph Metric Learning

Graph Metric Learning in PyTorch.

## Install

- Install (Virtualenv)[https://virtualenv.pypa.io/en/stable/].
    $ [sudo] pip install virtualenv

- Install all the requirements. Note that the following script will automatically creates a virtualenvironment, therefore if (Virtualenv)[https://virtualenv.pypa.io/en/stable/] is not installed, sudo is required.  
    $ [sudo] ./install.sh

## Usage

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

