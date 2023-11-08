# README

Simple CLI to send chat queries to ChatGPT.


## Run

Create the following script in order to easily run the CLI with openai models:

```bash
#!/usr/bin/env bash

export OPENAI_API_KEY=sk-your-openai-api-key
python3 -u ~/path/to/openai-cli/main.py $@
```

Or run any (local) API that mimics the OpenAI API and can be used as a drop-in replacement:

```bash
#!/usr/bin/env bash

export OPENAI_API_KEY=sk11111111
export OPENAI_API_BASE=http://127.0.0.1:5001/v1
python3 -u ~/path/to/openai-cli/main.py $@
```
