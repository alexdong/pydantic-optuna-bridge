.PHONY: test build release clean

UV := uv

SRC := src/pydantic_optuna_bridge

test:
	$(UV) run --with annotated-types --with pytest -m pytest

build: clean
	$(UV) build

release: clean
	$(UV) build
	$(UV) publish --token $${PYPI_TOKEN:?PYPI_TOKEN environment variable must be set}

clean:
	rm -rf dist build *.egg-info
