name: Bug report
description: Create a report to help us improve M23-LLM
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Please verify if the bug can be reproduced using the latest version of code on the `main` branch before submitting.
  - type: textarea
    id: describe-bug
    attributes:
      label: Describe the bug
      placeholder: A clear and concise description of what the bug is.
    validations:
      required: true
  - type: textarea
    id: reproduction-steps
    attributes:
      label: Steps to Reproduce
      placeholder: |
        1. Run train.py with arguments ...
        2. Set init_mode to 'm23'
        3. See error ...
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      placeholder: A clear and concise description of what you expected to happen.
    validations:
      required: true
  - type: textarea
    id: environment-details
    attributes:
      label: Environment details
      placeholder: |
        - OS: Windows 11 / Linux
        - PyTorch version: 2.7.1
        - GPU: RTX 4070 Ti Super
    validations:
      required: false
