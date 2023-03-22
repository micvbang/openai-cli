#!/usr/bin/env python3

import argparse
import sys
from itertools import zip_longest
from os import environ
from typing import Generator, Union

import openai
from openai.error import Timeout


def main(api_key: str, model: str, query_words: list, chat_timeout: int):
    openai.api_key = api_key

    query: str | None = None
    if len(query_words) > 0:
        query = " ".join(query_words)

    chat = start_chat_api(model, timeout=chat_timeout)
    next(chat)  # kick-start generator

    try:
        cli(chat, query=query)
    except (KeyboardInterrupt, EOFError):
        print("\nThanks, see you again!")


ChatResponse = Union[str, None, Timeout]


def cli(chat: Generator[ChatResponse, str, None], query: str | None):
    if query is None:
        query = input("Ask away!\nYou: ")
    else:
        sys.stdout.write(f"You: {query}\n\n")

    chat.send(query)

    sys.stdout.write("GPT: ")
    for msg_fragment in chat:
        is_timeout = isinstance(msg_fragment, Timeout)
        if msg_fragment is None or is_timeout:
            if is_timeout:
                sys.stdout.write("That took too long. Please try again.")
                next(chat)

            query = input("\n\nYou: ")
            chat.send(query)
            sys.stdout.write("GPT: ")

        if isinstance(msg_fragment, str):
            sys.stdout.write(msg_fragment)


def start_chat_api(model: str, timeout: int) -> Generator[ChatResponse, str, None]:
    queries: list[str] = []
    replies: list[str] = []

    query: str = yield None
    while True:
        yield None  # let caller throw away value returned in .send()
        queries.append(query)

        reply = ""

        try:
            chat = openai.ChatCompletion.create(
                model=model,
                messages=_make_messages(queries, replies),
                stream=True,
                request_timeout=timeout,
            )

            for msg in chat:
                delta = msg.choices[0].delta
                if "content" not in delta or delta.content == "\n\n":
                    continue

                yield delta.content
                reply += delta.content
        except Timeout as err:
            yield err

        # tell caller the current response has ended, and wait for new query
        query = yield None

        replies.append(reply)
        reply = ""


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
