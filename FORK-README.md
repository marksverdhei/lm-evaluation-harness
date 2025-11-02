This branch of my fork is for running evaluations locally personally.
I will be utilizing wandb to track the evaluations.

install:

```
uv venv
source .venv/bin/activate
uv sync
uv pip install -e ".[api, wandb]"
```
