# This code is completely untested.
import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys

from typing import List, Dict, Any, Tuple

import openai
from langchain import Model, register_model
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from error_prompter import ErrorPrompter


@register_model('openai')
class OpenAIModel(Model):
    def __init__(self, model_name_or_path, api_key):
        openai.api_key = api_key
        self.model_name_or_path = model_name_or_path

    def generate(self, prompt, num_responses=1, max_tokens=50, temperature=0.5, top_p=1):
        response = openai.Completion.create(
            engine=self.model_name_or_path,
            prompt=prompt,
            max_tokens=max_tokens,
            n=num_responses,
            temperature=temperature,
            top_p=top_p,
        )
        return [choice.text.strip() for choice in response.choices]


@register_model('transformers')
class TransformersModel(Model):
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def generate(self, prompt, num_responses=1, max_tokens=50, temperature=0.5, top_p=1):
        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        return generator(prompt, max_length=max_tokens, num_return_sequences=num_responses, temperature=temperature, top_p=top_p)


class ErrorPrompterWrapper:
    def __init__(self, model: Model, prompt_generator, confirm: bool):
        self.model = model
        self.prompt_generator = prompt_generator
        self.confirm = confirm

    def prompt(self, file_path: str, args: List[Any], error_message: str) -> str:
        with open(file_path, "r") as f:
            file_lines = f.readlines()

        file_with_lines = []
        for i, line in enumerate(file_lines):
            file_with_lines.append(str(i + 1) + ": " + line)
        file_with_lines = "".join(file_with_lines)

        initial_prompt_text = self.prompt_generator(file_path, args, error_message)

        prompt = (
            initial_prompt_text +
            "\n\n"
            "Here is the script that needs fixing:\n\n"
            f"{file_with_lines}\n\n"
            "Here are the arguments it was provided:\n\n"
            f"{args}\n\n"
            "Here is the error message:\n\n"
            f"{error_message}\n"
            "Please provide your suggested changes..."
        )

        if self.confirm:
            response = input(f"Are you sure you want to run the following code:\n\n{prompt}\n\n[Y/n] ")
            if response.lower() == "n":
                return ""

        suggestions = self.model.generate(prompt, num_responses=3, max_tokens=300, temperature=0.5, top_p=0.9)

        return suggestions[0]


def send_error_to_wrapper(file_path: str, args: List[Any], error_message: str, wrapper: ErrorPrompterWrapper) -> str:
    return wrapper.prompt(file_path, args, error_message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script that takes a file with a stacktrace and edits it using GPT-3')
    parser.add_argument('traceback_file', type=str, help='path to the file with the stacktrace')
    parser.add_argument('--api_key', type=str, required
    parser.add_argument('--input', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    traceback_file = args.traceback_file
    api_key = args.api_key
    if not api_key: api_key = envvir.load("OPENAI_API_KEY")
    input_file = args.input
    output_file = args.output
    verbose = args.verbose

    if verbose:
        print(f"Verbose logging is enabled. Input file: {input_file}. Output file: {output_file}.")
    else:
        print(f"Input file: {input_file}. Output file: {output_file}.")

if __name__ == '__main__':
    main()
