#!/bin/bash

# Clone LFS-tracked files
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
