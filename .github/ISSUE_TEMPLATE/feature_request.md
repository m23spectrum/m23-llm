name: Feature request / Research proposal
description: Propose mathematical improvements, architecture support or ideas for M23-LLM
labels: ["enhancement", "research"]
body:
  - type: textarea
    id: problem-relation
    attributes:
      label: Is your feature request related to a problem?
      placeholder: A clear and concise description of what the limitation is. (e.g. lack of adaptation for specific activation layers).
    validations:
      required: false
  - type: textarea
    id: solution-concept
    attributes:
      label: Describe the proposed solution
      placeholder: |
        Provide details on your mathematical idea, SVD spectrum change, group theory mapping, or new training regime (dLLM variants).
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you have considered
      placeholder: A clear and concise description of any alternative solutions or features you have considered.
  - type: textarea
    id: extra-context
    attributes:
      label: Additional Context
      placeholder: Add any other context or references (papers, formulas, code snippets) here.
