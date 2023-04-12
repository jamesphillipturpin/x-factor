# X-Factor
## About

The script x-factor.py is an extreme refactoring of the wolverine.py script for added functionality and a
higher level, object oriented programming (OOP) approach. The script is completely untested. The wolverine.py
script is preserved unchanged for now, so you can use it to improve the x-factor.py script is you wish. The
README for the wolverine.py script is included below.

# Wolverine

## About

Give your python scripts regenerative healing abilities!

Run your scripts with Wolverine and when they crash, GPT-4 edits them and explains what went wrong. Even if you have many bugs it will repeatedly rerun until it's fixed.

For a quick demonstration of the wolverine.py script see bio_bootloader's video [demo video on twitter](https://twitter.com/bio_bootloader/status/1636880208304431104).

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Add your openAI api key to `openai_key.txt` - _warning!_ by default this uses GPT-4 and may make many repeated calls to the api.

## Example Usage

To run with gpt-4 (the default, tested option):

    python wolverine.py buggy_script.py "subtract" 20 3

You can also run with other models, but be warned they may not adhere to the edit format as well:

    python wolverine.py --model=gpt-3.5-turbo buggy_script.py "subtract" 20 3

## Future Plans

This is just a quick prototype I threw together in a few hours. There are many possible extensions and contributions are welcome:

- add flags to customize usage, such as asking for user confirmation before running changed code
- further iterations on the edit format that GPT responds in. Currently it struggles a bit with indentation, but I'm sure that can be improved
- a suite of example buggy files that we can test prompts on to ensure reliablity and measure improvement
- multiple files / codebases: send GPT everything that appears in the stacktrace
- graceful handling of large files - should we just send GPT relevant classes / functions?
- extension to languages other than python
