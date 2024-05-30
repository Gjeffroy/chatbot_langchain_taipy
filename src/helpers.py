import json


def pretty_print_json(json_data):
    """Pretty print JSON data to the console.

    Args:
        json_data (dict): The JSON data to print.
    """
    print(json.dumps(json_data, indent=4))


def pretty_print_documents(documents):
    """Pretty print a list of documents to the console.

    Args:
        documents (list): A list of documents to print.
    """
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        pretty_print_json(doc.metadata)
        print(f"Content: {doc.page_content}")
        print("__" * 50)
        print("\n")