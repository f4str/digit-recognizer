# Digit Recognizer

Simple implementations of different types of neural network classifiers for the MNIST data set using PyTorch.

This serves as template starter code for setting up classifiers.

## Installation

Clone the repository.

```bash
git clone https://github.com/f4str/digit-recognizer
```

Change directories into the folder.

```bash
cd digit-recognizer
```

Install Python and create a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dev dependencies using pip.

```bash
pip install -e .[dev]
```

## Testing

The `tox` library is used to run all tests and code formatting. This is automatically installed with the dev requirements. The available options are as follows:

* Run linting checks using `flake8`

    ```bash
    tox -e lint
    ```

* Run type checks using `mypy`

    ```bash
    tox -e type
    ```

* Run unit tests `pytest`

    ```bash
    tox -e test
    ```

* Run all three of the tests above

    ```bash
    tox
    ```

* Format the code using `black` and `isort` to comply with linting conventions

    ```bash
    tox -e format
    ```

Upon pull request, merge, or push to the `master` branch, the three tests with `tox` will be run using GitHub Actions. The workflow will fail if any of the tests fail. See `.github/workflows/python-package.yml` for more information on how the CI works.
