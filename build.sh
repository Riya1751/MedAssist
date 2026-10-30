#!/usr/bin/env bash
set -o errexit

apt-get update
apt-get install -y tesseract-ocr libgl1

pip install --upgrade pip
pip install -r requirements.txt