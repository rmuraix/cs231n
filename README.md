# CS231n

## About

My personal notebooks for [CS231n](https://cs231n.stanford.edu/).

## Running Locally

If you have Docker installed, you can use devcontainer. Otherwise, the following are required.  

- OS that supports shell scripts
- Python3
- [uv](https://docs.astral.sh/uv/)

```sh
# Install dependencies (add `--group gpu` for torch)
uv sync

# Format code with ruff
uv run ruff format .

# Install pre-commit to check formatting
# and remove metadata from *.ipynb When you commit.
pre-commit install
```

## Course Materials

To see the course materials translated into Japanese, execute the following commands.

```sh
cd docs
docker compose up
```

Open http://localhost:3000

## References

- [Course page](https://cs231n.stanford.edu/index.html)
- [Assignments](https://cs231n.stanford.edu/assignments.html)

## Contributing

Your contribution is always welcome. Please read [Contributing Guide](https://github.com/rmuraix/.github/blob/main/.github/CONTRIBUTING.md).
