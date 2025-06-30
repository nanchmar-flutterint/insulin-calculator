import base64
import json
import os
from functools import lru_cache

import boto3

client = boto3.client("secretsmanager")

USERNAME_SECRET_ARN = os.environ["usernameSecretArn"]
PASSWORD_SECRET_ARN = os.environ["passwordSecretArn"]


@lru_cache(maxsize=1)
def get_auth_header():
    username = get_secret(USERNAME_SECRET_ARN, "username")
    password = get_secret(PASSWORD_SECRET_ARN, "password")
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


@lru_cache(maxsize=10)
def get_secret(secret_name: str, secret_key: str) -> str:
    response = client.get_secret_value(SecretId=secret_name)
    value = json.loads(response["SecretString"])
    return value[secret_key]


def lambda_handler(event: dict, __) -> dict:
    expected_auth_header = get_auth_header()
    is_authorized = (
        event.get("headers", {}).get("authorization") == expected_auth_header
    )

    if is_authorized:
        print(f"Successfully authenticated user")
    else:
        print(f"Could not authenticate user!")

    return {"isAuthorized": is_authorized}
