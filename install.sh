#!/bin/bash

pip3 install virtualenv

virtualenv -p python3 env
source env/bin/activate

pip install -r requirements.txt

deactivate

