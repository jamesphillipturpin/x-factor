# X-Factor
## About

Wolverine gave your python scripts regenerative healing abilities. X-Factor is like son of Wolverine, plus his team members. You get regenerative abilities with greater flxibility through more available resources.

The script x-factor.py is an extreme refactoring of the wolverine.py script for added functionality and a higher level, object oriented programming (OOP) approach. X-Factor does not rely entirely on one AI model or one curator of AI models. Instead, class wrappers are used to allow seemless access to both Application Program Interfaces (APIs) and direct downloads of models, such as models available from the Face Hugger community via Python's Transformer module. This gives you the flexibility to use AI products that meet your needs, such as security of proprietary and confidential information, and your preferences, such as a preference for open source software tools.

The x-factor.py script is experimental. What is provided is only a preliminary snapshot of an ongoing project.

## Call to Action

### Community Contribution
Contributors are encouraged to learn about the tools used to customize large language models. I am preparing and curating a YouTube playlist for those who want to immerse themselves.
https://youtube.com/playlist?list=PLluQwLgQf-EP2VJC20IqNLuzJKdPsSQjp

In the age of AI, we still need high level thinking and gumption to follow through on good ideas.
If you want to become a contributor to this repository, here are some ways you can help.

Here are some ways you can contribute to code, in order of most to least effort:
1. Make improvements that you have tested.
2. Make improvements that you haven't tested, but seem like good ideas. You may submit those as commented out code if you are afraid of breaking something.
3. Ask AI to make an improvement for you.

Here are some ways that you can contribute to the documentation, in no particular order:
1. Add an item to the Future Plans to-do list below.
2. Add documentation of existing features.
3. Improve the quality of the writing.
4. Ask AI to do any of the above.

### Consulting with the Repository Owner
If you want help with a similar or related project, please contact the author jamesphillipturpin. I took classes related to AI at Caltech, and I am also a licensed structural engineer, and experienced with algorithmic trading. If you have an AI related project that intersects one or more of those fields, it may be an especially good fit.

## Future Plans

Here is a to-do list. Besides you or I laboriously writing original code, anyone can also use the items on this list to prompt AI models for more changes to this or similar scripts. The pound symbol is intentionally employed as a double entendre to mean "number" and a comment code.

#1. Add docstrings and error-handling to the code, so that this program's code can itself be comprehended and edited easily by NLP modules.

#2. Adding flags to customize usage: Arg parsing has been implemented in the parse_args() function wrapper. However, you can add more command-line arguments to allow users to customize its behavior. For example, you can add a --confirm flag that asks the user for confirmation before running the edited code.

#3. Adding example buggy files: You can use existing datasets of example code, such as from the Face Hugger communitt, that cover a range of common errors. You can then use example files from datasets to test your prompts and measure their effectiveness.

#4. Handling multiple files / codebases: You can modify your script to accept multiple files as input. You can then parse the stacktrace to determine which files are relevant and send them to GPT-3 for editing.

#5. Handling large files: You can modify your script to only send relevant portions of the code to GPT-3 for editing. For example, you can only send the code within the function that caused the error. You can also experiment with using a language model like Langchain to manage memory.

#6. Supporting additional languages: You can modify your script to support additional programming languages. You will need to train GPT-3 on examples of code from the new language to ensure that it can provide accurate suggestions.

#7. Optimizing the code for speed and resource efficiency: You can optimize the code by minimizing the number of API calls made to GPT-3 and implementing more efficient data processing techniques. This can be done by using batches of files, implementing a caching system to store previous suggestions, and reducing the number of suggestions generated for each prompt.

#8. Adding a feature to track the changes made to the code: You can add a feature to track the changes made to the code by recording the original code and the edited code. This can be useful for debugging and for auditing the changes made to the code.

#9. Adding a feature to undo the changes made to the code: You can add a feature to undo the changes made to the code by saving a copy of the original file and the edited file. This can be useful in case the edited code causes more problems than it solves.

#10. Integrating with popular IDEs and text editors: You can create a plugin or extension for popular IDEs and text editors that integrates with the error prompter script. This can make it easier for developers to fix errors in their code without leaving their preferred development environment.

#11. Creating a web interface: You can create a web interface for the error prompter script that allows users to upload their code and receive suggestions for how to fix errors. This can be useful for developers who do not have access to GPT-3 or who prefer a more user-friendly interface.

#12. Adding a feature to suggest best practices: You can modify the prompt generator to suggest best practices for coding, such as using descriptive variable names or avoiding hard-coded values. This can help developers improve the quality of their code and reduce the likelihood of errors.

#13. Adding a feature to suggest alternative solutions: You can modify the prompt generator to suggest alternative solutions to the error that caused the prompt. This can be useful in cases where the original code is poorly designed or overly complex.

#14. Creating a dataset of human-written code corrections: You can create a dataset of human-written code corrections and use it to train a machine learning model to suggest code corrections. This can be used to supplement or replace the use of GPT-3 for error correction.

#15. Creating a suite of error correction tools: You can create a suite of error correction tools that can handle a wide range of programming languages and error types. This can be useful for developers who work with multiple languages or who encounter a variety of errors in their code.

#16. Creating a community-driven error correction platform: You can create a platform that allows developers to submit their code for review and correction by other developers. This can be useful for new developers who are looking to learn from more experienced developers, as well as for experienced developers who want to contribute to the community.

#17. Improving the training data for the language model: You can improve the training data for the language model by using a wider range of programming languages and code examples. This can help the model make more accurate suggestions and improve its overall performance.

#18. Adding a feature to automatically apply code corrections: You can modify the script to automatically apply the suggested code corrections to the original file. This can save developers time and reduce the likelihood of errors introduced by manual editing.

#19. Adding a feature to learn from the developer's corrections: You can modify the script to learn from the corrections made by the developer and use this information to improve the quality of its suggestions. This can help the script adapt to the developer's coding style and preferences.

#20. Adding a feature to suggest unit tests: You can modify the prompt generator to suggest unit tests for the code being edited. This can help developers ensure that their code is functioning correctly

# Wolverine

## Disclaimer

Although Wolverine offered some inspiration for this repository, Wolverine is deprecated. It is recommended that you not run the original wolverine.py file, as that could results in an infinite loop of expensive API calls. The wolverine.py file in this repository has had the head of the infinite loop commented out and replaced with a head that runs the loop once.

## About

Give your python scripts regenerative healing abilities!

Run your scripts with Wolverine and when they crash, GPT-4 edits them and explains what went wrong.

For a quick demonstration of the wolverine.py script see bio_bootloader's video [demo video on twitter](https://twitter.com/bio_bootloader/status/1636880208304431104).

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Add your openAI api key to `openai_key.txt` - _warning!_ by default this uses GPT-4.

## Example Usage

To run with gpt-4 (the default, tested option):

    python wolverine.py buggy_script.py "subtract" 20 3

You can also run with other models, but be warned they may not adhere to the edit format as well:

    python wolverine.py --model=gpt-3.5-turbo buggy_script.py "subtract" 20 3
