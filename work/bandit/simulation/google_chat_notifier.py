import requests
import json

webhook_url = "https://chat.googleapis.com/v1/spaces/AAAAOtt3NVo/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=VCp_BIdrvxi7WfdkEKwhul3H51GofwBZX9c8hsh7smM"

def send_message_to_google_chat(message):
    data = {"text": message}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
