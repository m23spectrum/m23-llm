# Contributing to M23-LLM

Thank you for your interest in contributing to M23-LLM! This project explores algebraic weight initialization (Mathieu Group M23 spectrum) and diffusion autoregressive training (dLLM).

## How to Contribute

### 1. Reporting Bugs
* Check the existing Issues to make sure the bug hasn't been reported.
* Use the **Bug Report** template to submit a new issue.
* Provide minimal code snippets or logs to help us reproduce the issue.

### 2. Suggesting Enhancements / Research Ideas
* Since this is an experimental research project, we highly welcome mathematical or algorithmic suggestions!
* Use the **Feature Request** template to propose modifications (e.g., adaptation to other LLM architectures, group actions, spectrum adjustments, fine-tuning scripts).

### 3. Submitting Pull Requests
* Fork the repository and create your branch from `main`.
* Ensure your code adheres to standard styling: run linters if applicable.
* If you modify the initialization algorithm in `m23_spectrum.py` or `m23_init.py`, run `compare_init.py` or test pre-training convergence to ensure no degradation in performance or gradient stability occurs.
* Fill out the Pull Request template completely.
