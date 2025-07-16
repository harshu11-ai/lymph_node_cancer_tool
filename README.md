
# 🧪 Lymph Node Cancer Tool Backend
This is a Flask-based backend service that allows doctors or researchers to upload histology images and receive predictions on whether the tissue shows signs of cancer.
It uses a fine-tuned ResNet18 PyTorch model trained on the PatchCamelyon dataset.

# 📋 Features
REST API endpoint /predict that accepts histology image uploads (.jpg, .jpeg, .png)
Returns a prediction (cancer or no cancer) with probability
Pre-trained ResNet18 model fine-tuned for cancer detection
Ready-to-deploy on platforms like Render, Heroku, or any cloud VM
Configured for use with Git LFS to handle large model files

# 📂 Project Structure
.
├── app.py                 # Flask backend server
├── cancer_classifier_best.pth  # Trained model checkpoint (tracked via Git LFS)
├── requirements.txt       # Python dependencies
├── .gitattributes         # Git LFS configuration
├── README.md              # This file

# 📦 Installation

Prerequisites
Python ≥ 3.8
Git LFS (for downloading the .pth model file)
pip / virtualenv

Clone & Setup
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/lymph_node_cancer_tool.git
cd lymph_node_cancer_tool

# Install Git LFS and pull the model file
git lfs install
git lfs pull

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# 📜 Notes
The model file (cancer_classifier_best.pth) is >100MB and tracked with Git LFS.
Make sure to install and initialize LFS before pulling the repo.
If deploying on free-tier platforms, consider optimizing or quantizing the model due to memory limits.

# 👨‍⚕️ License & Citation
This project is for research/educational purposes.
Please cite appropriately if you use it in your research.
