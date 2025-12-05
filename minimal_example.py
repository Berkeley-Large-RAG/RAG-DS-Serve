import requests
url = "http://api.ds-serve.org:30888/search"

headers = {"Content-Type": "application/json"}
payload = {
    # the first question in SimpleQA dataset
    "query": "Who received the IEEE Frank Rosenblatt Award in 2010?",
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()
print(result)