import requests

def get_pdf_urls(api_key, dataset_id):
    """
    Fetch the list of PDF URLs from the dataset.

    Args:
        api_key (str): The API key to authenticate the user.
        dataset_id (str): The dataset ID to access.

    Returns:
        list: A list of PDF URLs.
    """
    endpoint = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"

    # Make a GET request to the API
    response = requests.get(endpoint, headers={"X-API-KEY": api_key})
    response = response.json()

    # Extract resource URLs
    resource_urls = {resource['format']: resource["url"] for resource in response["resources"]}

    # Download the JSON data
    data_json = requests.get(resource_urls["json"]).json()

    # Extract PDF URLs and format their names
    pdf_urls = [
        {
            'discipline': pdf["discipline"],
            'cycle': pdf["niveau_d_enseignement"],
            'description': pdf["descriptif"],
            'url': pdf["contenu_sur_le_site"],
            'doc_id': 'd'+ str(i), # Add a unique ID for each document to be used in the retriever (full URL makes the prompt too long)
        }
        for i, pdf in enumerate(data_json)]

    return pdf_urls
