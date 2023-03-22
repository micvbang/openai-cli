#!/usr/bin/env python3

import argparse
import sys
from itertools import zip_longest
from os import environ
from typing import Generator

import openai
from openai.error import Timeout


def main(api_key: str, model: str, query_words: list, chat_timeout: int):
    openai.api_key = api_key

    query: str | None = None
    if len(query_words) > 0:
        query = " ".join(query_words)

    chat = ChatGPT(model=model, timeout=chat_timeout)

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
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--api-key", default=environ.get("OPENAI_API_KEY"))
    parser.add_argument("--chat-timeout", default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        api_key=args.api_key,
        model=args.model,
        query_words=args.query,
        chat_timeout=args.chat_timeout,
    )
