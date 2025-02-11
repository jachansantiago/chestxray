# ChestXRay

A deep learning project for automatic pneumonia detection using chest X-ray images.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project evaluates deep learning models for automatic pneumonia detection. We compare the performance of models trained from scratch with those utilizing transfer learning from general datasets like ImageNet. Key metrics reported include sensitivity, specificity, and overall accuracy. Additionally, we provide Class Activation Maps (CAMs) to illustrate model predictions.

## Features

- Comparison of transfer learning and training from scratch
- Evaluation using sensitivity, specificity, and accuracy metrics
- Visualization of Class Activation Maps for model interpretability

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jachansantiago/chestxray.git
   cd chestxray
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

To train the models, execute the following script:

```bash
./run_exp.sh
```

### Generating Plots

Utilize the provided [visualization notebook](https://github.com/jachansantiago/chestxray/blob/main/visualizations.ipynb) to:

- Plot training and validation losses
- Visualize Class Activation Maps
- Display sample training data

## Results

Our experiments indicate that while transfer learning can enhance overall model accuracy, training from scratch yields higher sensitivity, which is crucial for medical diagnostics. Detailed results and visualizations are available in the repository.

## Contributing

We welcome contributions to improve this project. Please fork the repository, create a new branch for your feature or bug fix, and submit a pull request for review.

## License

This project is licensed under the [MIT License](LICENSE).
