# X-Factor
## Overview

X-Factor is a Python script that provides an object-oriented approach to programming with regenerative healing abilities. It builds on the functionality of the Wolverine script while introducing more flexibility and accessibility through the use of class wrappers. This allows seamless access to both Application Program Interfaces (APIs) and direct downloads of AI models, including those available from the Face Hugger community via Python's Transformer module. X-Factor is designed to provide the flexibility needed to use AI products that meet your needs, whether that be for security of proprietary information or a preference for open-source software tools.

## Setup

    git clone https://github.com/jamesphillipturpin/x-factor.git
    cd x-factor
    pip install requirements.txt

Add your openAI api key to `openai_key.txt`.

## Example Usage

To run:

    python x-factor.py buggy_script.py --args="Add 1 2"

### Status
Please note that the x-factor.py script is still experimental, and what is provided is only a preliminary snapshot of an ongoing project. However, it does have added features:

1. To avoid wasting resources of unnecessary repeat runs, both the Traceback and the messages from NLP models are saved in separate files.
2. To avoid potential data loss from automated processes, files are backed up before being overwritten.
3. For maximum flexibility, API keys are checked in both a local text file and in environmental variables.

### Coming Soon / Partially Implemented:

* Specify which NLP model to use.
* Flexible choices for prompt templates.

## How to Contribute

We welcome contributors to join our community and help improve the X-Factor project. There are several ways you can contribute, including:
Code Contributions

If you have experience in programming and want to help, there are several ways you can do so. Here they are listed from most to least effort:

 1. Test and make improvements to the code.
 2. Submit improvements as commented-out code if you're afraid of breaking anything.
 3. Use AI to suggest improvements.

### Documentation Contributions

If you're interested in contributing to the documentation of the project, here are a few ways to help:

1. Add items to the Future Plans to-do list (listed below).
2. Document existing features.
3. Improve the quality of the writing.
4. Ask AI to do any of the above.

### Consulting with the Repository Owner

If you're interested in starting a similar project or need help with a related project, please contact the author jamesphillipturpin. He has a background in AI, is a licensed structural engineer, and has experience with algorithmic trading. If your project intersects with one or more of these fields, it may be an especially good fit.
### Future Plans

Here is a list of to-do items for future versions of X-Factor. Anyone is welcome to use these items to prompt AI models for more changes to this or similar scripts.

1. Add docstrings and error-handling to the code so that it can be easily comprehended and edited by NLP modules.
2. Add flags to customize usage, such as a --confirm flag that asks the user for confirmation before running the edited code.
3. Add example buggy files to test the effectiveness of prompts.
4. Handle multiple files/codebases by modifying the script to accept multiple files as input and parse the stacktrace to determine which files are relevant.
5. Handle large files by modifying the script to only send relevant portions of the code to GPT-3 for editing.
6. Support additional programming languages by training GPT-3 on examples of code from the new language.
7. Optimize the code for speed and resource efficiency.
8. Add a feature to track the changes made to the code.
9. Add a feature to undo the changes made to the code.
10. Create a plugin or extension for popular IDEs and text editors that integrates with the error prompter script.
11. Create a web interface for the error prompter script that allows users to upload their code and receive suggestions for how to fix errors.
12. Prompt Soft Tuning
13. Few Shot Model Fine Tuning

Thank you for your interest in X-Factor. We look forward to your contributions!


