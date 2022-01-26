## Modular Search Engine

A generic modularised search engine for testing and experimenting with multiple information retrieval techniques.
Comes with a HTML document cleaner module.

### Requirement

1.  Python 3
2.  Pipenv (optional but recommended)

### Setup

Add documents to corpus folder.

To setup the application run the following commands.
Use the `./run boolean` for a Boolean model engine (default if flag is omitted).
Use the `./run vector` for a Boolean model engine (default if flag is omitted).

```
cd search_engine_cli
mkdir index
mkdir corpus
pipenv install
pipenv shell
chmod +x run
./run
```

### For
