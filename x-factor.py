"""
Example usage: Type or past the following line to the command prompt:
python x-factor.py buggy_script.py --args="Add 1 2"
"""

"""
There are several functions and classes in the provided code. Here is a brief description of each:
parse_args(): This function parses the command-line arguments and returns them as a Namespace object.
get_default_error_message_traceback(script): This function takes a script file path and returns a default error message traceback file path. It uses the script filename to create a temporary error message traceback file path in the same directory as the script.
get_default_revision(script): This function takes a script file path and returns a default revision file path. It uses the script filename to create a temporary revision file path in the same directory as the script.
split_path(path): This function takes a file path and returns a tuple containing the directory name and file name components.
OpenAI_API_Model: This class represents an OpenAI model that can generate responses to prompts. It has a generate() method that takes a prompt, number of responses, maximum tokens, temperature, and top p as arguments and returns the generated responses.
TransformersModel: This class represents a Hugging Face model that can generate responses to prompts. It has a generate() method that takes a prompt, number of responses, maximum tokens, temperature, and top p as arguments and returns the generated responses.
run_script(script_filepath: str, script_args_as_string: str): This function runs a script using the subprocess module of Python and returns its output.
The script reads a file with a stack trace and applies a NLP model to suggest fixes to the code. It prompts the user with the error message, the file path, and a list of arguments passed to the script, and they can provide suggested changes to the file. The script uses two models: OpenAI and Transformers. The OpenAI model generates the text prompts and the Transformers model generates the responses. The script parses command-line arguments and uses the argparse module to do so. It also reads the contents of the stack trace file and creates a list of argument strings.
"""

# Import statements are given in alphabetical order to make organization uniquely defined.
import argparse
import os
import openai
import shutil
import subprocess
import sys
from langchain import LLMChain, OpenAI as openai_langchain, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI as openai_llm
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from termcolor import cprint
from typing import Any, Dict, List, Optional, Tuple

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments. Define defaults so most arguments may be convenitently omitted.

    Returns:
        A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Script that takes a file with a stacktrace and edits it using GPT-3')
    parser.add_argument('script', type=str, help='script file path')
    parser.add_argument('--args', default="", type=str, help='arguments to be used when running script')
    parser.add_argument('--openai_api_key', default=os.environ.get('OPEN_AI_API_KEY'), help='API key for OpenAI')
    parser.add_argument('--error_message_traceback', default=None, type=str, help='error_message_traceback_file filepath to save output and/or messages from running script')
    parser.add_argument('--revision', default=None, type=str, help='script revision file path')
    parser.add_argument('--verbose', action='store_true', default=False,  help='enable verbose logging')
    parser.add_argument('--confirm', action='store_true', default=False, help="Prompts the user for confirmation before applying the suggested fix.")
    parser.add_argument('--model', default="code-davinci-002", help="Name or path of the model.")
    parser.add_argument('--hf', action='store_true', default=False, help="Use a Hugging Face model.")
    args = parser.parse_args()
    if args.error_message_traceback is None:
        args.error_message_traceback = get_default_error_message_traceback(args.script)
    if args.revision is None:
        args.revision = get_default_revision(args.script)
    print()
    print("Results of argument parser are printed below:")
    print(args)
    print()
    return args

def get_default_error_message_traceback(script):
    dirname, script_filename = os.path.split(script)
    error_message_traceback_filename = f"{script_filename}.x-factor.error_message_traceback.temporary.txt"
    error_message_traceback_filepath = os.path.join(dirname,  error_message_traceback_filename)
    return error_message_traceback_filepath

def get_default_revision(script):
    dirname, script_filename = os.path.split(script)
    revision_filename = f"{script_filename}.x-factor.revision.temporary.txt"
    revision_filepath = os.path.join(dirname,  revision_filename)
    return revision_filepath

def split_path(path):
    """
    Split a file path into its directory name and file name components.

    Args:
        path (str): The path to split.

    Returns:
        A tuple containing the directory name and file name.
    """
    dirname, filename = os.path.split(path)
    return dirname, filename


class OpenAI_API_Model(object):
    def __init__(self, model_name_or_path="code-davinci-002"):
        self.model = openai_llm(model_name = model_name_or_path)
        self.chat = ChatOpenAI()

    def generate(self, prompt:dict, num_responses=1, max_tokens=50, temperature=0.5, top_p=1):
      try:
        llm_chain = LLMChain(prompt=prompt, llm=self.model)
        chain = LLMChain(llm=self.model, prompt=prompt)
      except:
        print()
        print("Printing prompt")
        print(prompt)
        print()
        print("Printing self.model")
        print(self.model)
        print()
        print("Printing llm_chain")
        print(llm_chain)
        print()
      finally:
        output = llm_chain.run()
        return output

class TransformersModel(object):
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def generate(self, prompt, num_responses=1, max_tokens=50, temperature=0.5, top_p=1):
        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        return generator(prompt, max_length=max_tokens, num_return_sequences=num_responses, temperature=temperature, top_p=top_p)

# Run a script and return its output
def run_script(script_filepath: str, script_args_as_string: str):
    script_args_as_tuple = (tuple(script_args_as_string.split(" ")))
    try:
        result = subprocess.check_output(
            [sys.executable, script_filepath, *script_args_as_tuple], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8"), e.returncode
    return result.decode("utf-8"), 0

def make_prompt(prompt: str, code: str, error_message_traceback: str, dialogue_history: Optional[str] = None) -> dict:
    #prompt=PromptTemplate(
    #    template = "Prompt:\n{prompt}\n\nCode:\n{code}\n\nError Message Traceback:\n{error_message_traceback}",
    #    input_variables=["prompt", "code", "error_message_traceback"],
    #)
    input_variables=["prompt", "code", "error_message_traceback"]
    template="Prompt:\n{prompt}\n\nCode:\n{code}\n\nError Message Traceback:\n{error_message_traceback}"
    template_args = (prompt, code, error_message_traceback)
    template_kwargs = {key: arg for (key, arg) in zip(input_variables, template_args)}
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template,
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt])
    print()
    print ("system_message_prompt")
    print (system_message_prompt)
    print()
    print ("chat_prompt_template")
    print (chat_prompt_template)
    print()
    return chat_prompt_template, template_args, template_kwargs



def save_to_file(text: str, filepath: str):
  try:
    """
    This is a function that writes a given text to a specified filepath. If a file already exists at the specified filepath, the function will create a backup file with an incrementing suffix before overwriting the original file.
    It first checks whether a file already exists at the specified filepath using the os.path.exists() function.
    If a file already exists, it creates a backup file with an incrementing suffix. For example, if filepath is "example.txt" and a backup file already exists with the name "example.txt.bak1", it will create a new backup file with the name "example.txt.bak2", and so on.
    It then opens the original file (filepath) in read mode and the backup file in write mode using the with open() context manager.
    It reads the contents of the original file using f1.read() and writes them to the backup file using f2.write().
    If a backup file was created, it is closed automatically when the with block ends.
    The function then opens the original file (filepath) in write mode using the with open() context manager.
    It writes the specified text to the file using f1.write().
    The file is closed automatically when the with block ends.
    """
    print()
    print(filepath)
    print(text)
    print(filepath)
    print()
    if os.path.exists(filepath):
        backup_suffix = 1
        backup_filepath = f"{filepath}.bak{backup_suffix}"
        while os.path.exists(backup_filepath):
            backup_suffix += 1
            backup_filepath = f"{filepath}.bak{backup_suffix}"
        with open(filepath, "r") as f1, open(backup_filepath, "w") as f2:
            f2.write(f1.read())
    
    with open(filepath, "w") as f1:
        f1.write(text)
  except Exception as e:
    print()
    print("Printing filepath")
    print(filepath)
    print()
    raise e

def main():
    args = parse_args()
    script_filepath = args.script
    script_args = args.args
    error_message_traceback_filepath = args.error_message_traceback
    revision_filepath = args.revision
    openai_api_key = args.openai_api_key
    openai.api_key = openai_api_key
    verbose = args.verbose
    model_name_or_path = args.model
    isHuggingFace = args.hf

    if isHuggingFace:
        model = TransformersModel(model_name_or_path=model_name_or_path)
    else:
        model = OpenAI_API_Model(model_name_or_path=model_name_or_path)

    if verbose:
        print(f"Verbose logging is enabled. Input file: {script_filepath}. Arguments: {args}. Traceback file: {error_message_traceback_filepath}.")
    else:
        print(f"Input file: {script_filepath}. Arguments: {args}. Traceback file: {error_message_traceback_filepath}.")

    output, returncode = run_script(script_filepath, script_args)    

    max_trials = 1
    dialogue_history = list()
    instruction = "Please improve the code based on the error messages and the intent of the code."
    for _ in range(max_trials):
        error_message_traceback, returncode = run_script(script_filepath, script_args)
        save_to_file(error_message_traceback, error_message_traceback_filepath)
        if returncode == 0:
            cprint("Script ran successfully.", "blue")
            print("Script Dialogue:", error_message_traceback)
            break
        else:
            cprint("Script crashed. Trying to fix...", "red")
            print("Error message:", error_message_traceback)

            chat_prompt_template, template_args, template_kwargs = make_prompt(
                     instruction,
                     script_filepath, 
                     error_message_traceback_filepath,
                     dialogue_history
            ) 
            print()
            print("Printing chat_prompt_template")
            print(chat_prompt_template)
            print()
            #chain = LLMChain(llm=model, prompt=chat_prompt_template)
            chat = ChatOpenAI(temperature=0.9)
            chain = LLMChain(llm=chat, prompt=chat_prompt_template)
            revision = chain.run(template_kwargs)
        save_to_file(revision, revision_filepath)
        dialogue_history.append(chat_prompt_template)
        dialogue_history.append(revision)

        if "Error" not in error_message_traceback: break

if __name__ == '__main__':
    main()
