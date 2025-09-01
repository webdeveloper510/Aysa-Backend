from django.test import TestCase

import requests

try:
    url ="https://api.the-aysa.com/admin-login"

    headers = {
        "content_type": "application/json"
    }

    data = {
        "password" : "admin@#098$!"
    }


    response = requests.post(url , headers=headers , json=data)
    print("response code : \n ", response.status_code)
    print()
    print("response text : \n ", response.text)

except Exception as e:
    print("Error is ", str(e))
