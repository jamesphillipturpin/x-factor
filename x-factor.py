import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
from os import environ

from typing import List, Dict, Any, Tuple

import openai
from langchain.llms.openai import OpenAI
from langchain import PromptTemplate, LLMChain

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from error_prompter import ErrorPrompter

@register_model('openai')
class OpenAIModel(Model):
    def __init__(self, model_name_or_path="code-davinci-002"):
        self.model = OpenAI(model_name = model_name_or_path)

    def generate(self, prompt, num_responses=1, max_tokens=50, temperature=0.5, top_p=1):
        llm_chain = LLMChain(prompt=prompt, llm=self.model)
        output = llm_chain.run()
        return output


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
        """
        Prompt the user with an error message, a file path, and a list of arguments,
        and return the suggested changes to the file.

        Args:
            file_path (str): Path to the file with the error.
            args (List[Any]): List of arguments passed to the script.
            error_message (str): Error message.

        Returns:
            str: Suggested changes to the file.
        """
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
    """
    Parse command-line arguments.

    Returns:
        A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Script that takes a file with a stacktrace and edits it using GPT-3')
    parser.add_argument('traceback_file', type=str, help='path to the file with the stacktrace')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--input', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    parser.add_argument('--confirm', action='store_true', default=False, help="Prompts the user for confirmation before applying the suggested fix.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    traceback_file = args.traceback_file
    api_key = args.api_key
    if not api_key: api_key = envvir.load("OPENAI_API_KEY")
    openai.api_key = api_key
    input_file = args.input
    output_file = args.output
    verbose = args.verbose

    if verbose:
        print(f"Verbose logging is enabled. Input file: {input_file}. Output file: {output_file}.")
    else:
        print(f"Input file: {input_file}. Output file: {output_file}.")

    with open(traceback_file) as f:
        traceback_lines = f.readlines()

    args_list = [input_file, output_file] if input_file and output_file else []
    error_prompter_wrapper = ErrorPrompterWrapper(OpenAIModel(api_key), ErrorPrompter(prompt_generator=stacktrace_prompt_generator), args.confirm)

    for line in traceback_lines:
        error_message = line.strip()
        file_path, line_number = get_file_path_and_line_number(error_message)
        if not file_path:
            continue

        file_path = os.path.abspath(file_path)
        args_string = line.split(file_path)[1].strip()
        args_list += parse_args_string(args_string)

        suggested_changes = send_error_to_wrapper(file_path, args_list, error_message, error_prompter_wrapper)
        if suggested_changes:
            if args.confirm:
                print(f"Suggested changes: {suggested_changes}")
            else:
                apply_suggested_changes(suggested_changes, file_path)

        args_list = [input_file, output_file] if input_file and output_file else []


if __name__ == '__main__':
    main()
