## Getting Started

This project creates a visualization of a VAE-trained network on ECG data. 

### Prerequisites

First thing you will need is the CPSC2018 Training dataset here https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mccabe-david/vae-visualization.git
   ```
2. Go inside the project's root file, and install required packages
   ```sh
   pip install -r requirements.txt
   ```

After this, the only thing left to do is running the project with

```sh
python vae.py
```