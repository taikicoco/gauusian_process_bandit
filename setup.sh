#!/bin/bash

python3.9 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r ./Docker/requirements.txt
