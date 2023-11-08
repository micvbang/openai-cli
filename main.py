#!/usr/bin/env python3

import argparse
import sys
from itertools import zip_longest
from os import environ
from typing import Generator

import openai
from openai.error import Timeout

DEFAULT_OPENAI_BASE_API = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"

def main(api_key: str, api_base: str,
        model: str, query_words: list, chat_timeout: int):
    """
    Initializes OpenAI API by setting up a ChatGPT object and starts the CLI loop.

    Args:
        api_key (str): An optional API key for authentication, 
            required to be meaningful for offical openai only.
        api_base (str): Base URL of the API endpoint, 
            can point to local openai compatible API.
        model (str): Name of the model to use, 
            required for offical openai only, otherwise irrelevant.
        query_words (list): List of words used to start the conversation.
        chat_timeout (int): Timeout duration for each API call.

    Raises:
        KeyboardInterrupt: If the user interrupts the program with Ctrl+C or EOF.
    """
    
    openai.api_key = api_key
    openai.api_base = api_base
    
    chat = ChatGPT(model=model, timeout=chat_timeout)

    query: str | None = None
    if len(query_words) > 0:
        query = " ".join(query_words)

    try:
        cli(chat, query=query)
    except (KeyboardInterrupt, EOFError):
        print("\nThanks, see you again!")

class ChatGPT:
    def __init__(self, model: str, timeout: int):
        self._model = model
        self._timeout = timeout
        self._queries: list[str] = []
        self._replies: list[str] = []

    def ask(self, query: str) -> Generator[str, None, None]:
        self._queries.append(query)

        chat = openai.ChatCompletion.create(
            model=self._model,
            messages=_make_messages(self._queries, self._replies),
            stream=True,
            request_timeout=self._timeout,
        )

        reply = ""
        for msg_fragment in chat:
            delta = msg_fragment.choices[0].delta
            if "content" not in delta or delta.content == "\n\n":
                continue

            yield delta.content
            reply += delta.content
        self._replies.append(reply)


def cli(chat: ChatGPT, query: str | None):
    if query is None:
        query = input("Ask away!\nYou: ")
    else:
        sys.stdout.write(f"You: {query}\n\n")

    while True:
        sys.stdout.write("GPT: ")

        message_fragments: Generator[str, None, None] | None = None
        try:
            message_fragments = chat.ask(query)
        except Timeout:
            sys.stdout.write("That took too long. Please try again.")

        if message_fragments:
            for message_fragment in message_fragments:
                sys.stdout.write(message_fragment)

        query = input("\n\nYou: ")


def _make_messages(queries: list[str], replies: list[str]) -> list[dict]:
    messages = []

    for query, reply in zip_longest(queries, replies):
        if query is not None:
            messages.append({"role": "user", "content": query})
        if reply is not None:
            messages.append({"role": "assistant", "content": reply})
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", default=[])
    parser.add_argument("--model", default=environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=environ.get("OPENAI_API_BASE", DEFAULT_OPENAI_BASE_API))
    parser.add_argument("--chat-timeout", default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        query_words=args.query,
        chat_timeout=args.chat_timeout,
    )
