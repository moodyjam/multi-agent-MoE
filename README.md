# Multi-Agent Mixture of Experts
An implemention of the Mixture of Experts architecture for Multi-Agent Systems. For a baseline comparison, we also use a custom Pytorch Lightning Implementation of the Distributed Neural Network Optimization for Multi-Robot Collaborative Learning algorithm ([Link to videos and Github](https://msl.stanford.edu/projects/dist_nn_train)).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10.6 installed
- A virtual environment manager (e.g., `venv`, `conda`)

## Installation

To install the necessary dependencies and set up the environment, follow these steps:

1. Create and activate a virtual environment:

   For `venv`:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   For `conda`:
   ```
   conda create --name myenv python=3.10.6
   conda activate myenv
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run multi-agent mixture of experts, navigate to the project directory and execute the `run.py` file with the necessary arguments. For example:

```
python run.py --config=configs/multi_agent_moe_4_datasets.yaml
```

Replace `configs/configs/multi_agent_moe_4_datasets.yaml` with the path to with a custom configuration file if different.

To run DiNNO, navigate to the project directory and execute `run.py` file with the following configuration:

```
python run.py --config=configs/dinno_4_datasets.yaml
```


