import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
webhook_url = os.getenv("WEBHOOK_URL")

def send_message_to_google_chat(message):
    data = {"text": message}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
