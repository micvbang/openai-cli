#!/usr/bin/env python3

import argparse
import sys
from itertools import zip_longest
from os import environ
from typing import Generator

from openai import OpenAI, APITimeoutError


def main(api_key: str, model: str, query_words: list,):
    query: str | None = None
    if len(query_words) > 0:
        query = " ".join(query_words)

    chat = ChatGPT(api_key=api_key, model=model)

    try:
        cli(chat, query=query)
    except (KeyboardInterrupt, EOFError):
        print("\nThanks, see you again!")


class ChatGPT:
    def __init__(self, api_key: str, model: str):
        self._model = model
        self._queries: list[str] = []
        self._replies: list[str] = []
        self._client = OpenAI(api_key=api_key)

    def ask(self, query: str) -> Generator[str, None, None]:
        self._queries.append(query)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=_make_messages(self._queries, self._replies),
            stream=True,
        )

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            yield delta
            reply += delta
        self._replies.append(reply)


def cli(chat: ChatGPT, query: str | None):
    if query is None:
        query = input("Ask away!\nYou: ")
    else:
        sys.stdout.write(f"You: {query}\n\n")

    while True:
        sys.stdout.write("\nGPT: ")

        message_fragments: Generator[str, None, None] | None = None
        try:
            message_fragments = chat.ask(query)
        except APITimeoutError:
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
    parser.add_argument("--model", default=environ.get("OPENAI_MODEL", "gpt-4-0125-preview"))
    parser.add_argument("--api-key", default=environ.get("OPENAI_API_KEY"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        api_key=args.api_key,
        model=args.model,
        query_words=args.query,
    )
