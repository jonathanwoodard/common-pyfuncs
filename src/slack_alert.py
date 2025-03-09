import os
from datetime import datetime as dt, timedelta as td
from slack_sdk.webhook import WebhookClient
from slack_sdk.errors import SlackApiError

"""
Use slack_sdk to send messages
Create a webhook in slack and save it in a hidden .slack directory
https://api.slack.com/messaging/webhooks
"""


def slack_alert(msg,lvl):
    # you can add emojis and/or gifs in the payload
    stamp = dt.strftime(dt.now(),"%Y-%m-%d %H:%M:%S")
    levels = {'low':[':facepalm:',"Phillip","https://media.giphy.com/media/ANbD1CCdA3iI8/giphy.gif"]
              ,'medium':[':scream:',"Bender","https://media.giphy.com/media/tNC2rod1uTrdC/giphy.gif"]
              ,'high':[':shitdisco:',"Fixit","https://media.giphy.com/media/KqWzEMydtRHX2/giphy.gif"]
              ,'critical':[':poop_fire:',"Doom","https://media.giphy.com/media/L18eMUGDk3vcwOPUGw/giphy.gif"]}
    try:
        with open('.slack/url') as f:
            url = f.read()
    except FileNotFoundError as e:
        print(e)
        return
    webhook = WebhookClient(url)
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": levels[lvl][0],
                    "emoji": True
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "image",
                "title": {
                    "type": "plain_text",
                    "text": "short error message",
                    "emoji": True
                },
                "image_url": levels[lvl][2],
                "alt_text": levels[lvl][1]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{stamp}\t*{msg.strip()}*"
                }
            },
            {
                "type": "divider"
            }
        ]
    }
    r = webhook.send(blocks=payload['blocks'])
    return r
