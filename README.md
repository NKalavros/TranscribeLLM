# Do you summarize papers with LLMs? Want to also add a talk? [WIP]

Well, how to they do? I've built a quick evaluation platform with rankings and scores to gather data from.

Just upload the PDF and it serves answers which you can rate. Dependencies are kept to an absolute minimum.

## Current supported models (I should really add icons here):

GPT4o

Deepseek

Claude

Gemini

Perplexity

Llama3

Grok2

More to come

## Installation

### Clone the repo:

```https://github.com/NKalavros/PaperLLM```

### Install the reqs:

```sudo apt-get install redis```

```pip install -r requirements.txt```

### Create the .env file in the same repository:

```
OPENAI_API_KEY="***"
DEEPSEEK_API_KEY="***"
CLAUDE_API_KEY="***"
GEMINI_API_KEY="***"
PERPLEXITY_API_KEY="***"
LLAMA_API_KEY="***"
GROQ2_API_KEY="***"
```

### Run

For this one, I usually keep 3 separate terminals (It's a dev project, don't come after me).

#### Terminal 1 (DB logging):

```
redis-server
```

#### Terminal 2 (Task management):

```
celery -A app.celery worker --loglevel=info --without-heartbeat --without-mingle
```

#### Terminal 3 (Actual app):

```
python app.py
```

## Visualization

I also have a rudimentary R Shiny app to go with this project and a debugging script that generates fake requests for the json. 

## Features TBD:

1. Add better PDF summarization (based on PaperQA2).
2. Add cost estimates for each model
3. VPS upload script
4. Side-project for an automated benchmarking based on existing structured query answer pairs based on existing database
5. Local running for models that can be used. Probably DeepSeek.
6. Difficulty of question (Easy, Medium, Hard)


### Notes:

https://devavrat.mit.edu/wp-content/uploads/2017/11/Iterative-ranking-from-pair-wise-comparisons.pdf
Effectively we need ELO
