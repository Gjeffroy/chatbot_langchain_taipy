import json
from langchain_core.documents.base import Document

def pretty_print_json(json_data, indent=4):
    """Pretty print JSON data to the console.

    Args:
        json_data (dict): The JSON data to print.
    """
    print(json.dumps(json_data, indent=indent, ensure_ascii=False))


def pretty_print_documents(documents, indent=2):
    """Pretty print a list of documents to the console.

    Args:
        documents (list): A list of documents to print.
        content_indent (int): The number of times to double the indent for content.
    """
    print("__" * 50)
    for i, doc in enumerate(documents):
        print('\033[94m' + f"Document {i + 1}:" + '\033[0m')
        print('\033[93m' + "Metadata:" + '\033[0m')
        pretty_print_json(doc.metadata, indent=1)
        print('\033[93m' + "Content:" + '\033[0m')
        content_lines = doc.page_content.split('\n')
        for line in content_lines:
            print(f"{' ' * 4}{line}")
        print("__" * 50)
        print("\n")


def pretty_print_chain_results(d, indent=0):
    """
    Pretty prints a dictionary with custom formatting.

    Parameters:
    d (dict): The dictionary to be pretty printed.
    indent (int): The current indentation level.

    Returns:
    None
    """
    for key, value in d.items():
        # Print the key with the current level of indentation
        print('\033[93m' + str(key) + ':\033[0m', end=' ')

        if isinstance(value, dict):
            # If the value is a dictionary, print newline and recursively print the dictionary
            print()
            pretty_print_chain_results(value, indent=1)
        elif isinstance(value, list):
            # If the value is a list, handle the list elements
            if value and isinstance(value[0], Document):
                print()
                pretty_print_documents(value, indent=indent)
            else:
                print('[')
                for item in value:
                    if isinstance(item, dict):
                        pretty_print_chain_results(item, indent + 1)
                    else:
                        print('    ' * (indent + 1) + str(item) + ',')
                print('    ' * indent + ']')
        else:
            # Print the value
            print(str(value))