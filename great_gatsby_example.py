import asyncio
import os
import anthropic
import requests
import json
from jsonformer_claude.main import JsonformerClaude

api_key = os.environ["ANTHROPIC"]
client = anthropic.Anthropic(api_key=api_key)
GENERATION_MARKER = "|GENERATION|"


def main():
    # Fetch "The Great Gatsby" text from the URL
    # url = "http://gutenberg.net.au/ebooks02/0200041.txt"
    # response = requests.get(url)
    # text = response.text

    text = "Spot is a little brown dog who lives in Stuytown with his family, Jack & Angie. He loves to play with his toys and go for walks in the park. He is a very good"

    gen_json = JsonformerClaude(
        anthropic_client=client,
        max_tokens_to_sample=2000,
        model="claude-v1",
        json_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        },
        prompt=f"Generate names and descriptions for all characters, even minor ones in the following book:\n {text}",
        debug=True,
    )

    print(json.dumps(gen_json(), indent=2))


if __name__ == "__main__":
    main()
