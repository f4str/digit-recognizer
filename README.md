# Digit Recognizer

Simple implementations of different types of neural network classifiers and training scripts for the MNIST data set using `PyTorch`. Also includes a GUI using `Tkinter` to draw digits and predict accordingly.

## Installation

Clone the repository.

```bash
git clone https://github.com/f4str/digit-recognizer
```

Change directories into the cloned repository.

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

## Usage

### Model Training

First train a model with the desired name and arguments.

```bash
python digit_recognizer/train.py --name {NAME} [...args]
```

This will create and train a model in `saved_models/{NAME}` based on the arguments provided. To view information on the available arguments, pass the `-h` flag.

### Model Evaluation

A trained model can be evaluated if desired. Ensure the name and arguments matches the saved model exactly.

```bash
python digit_recognizer/test.py --name {NAME} [...args]
```

To view information on the available arguments, pass the `-h` flag.

### GUI Canvas

Launch the GUI which will create a canvas to draw and recognize digits using the saved model. Ensure the name and arguments matches the saved model exactly.

```bash
python digit_recognizer/gui.py --name {NAME} [...args]
```

To view information on the available arguments, pass the `-h` flag.

In the canvas, draw by clicking and holding left click. To erase any segment, click and hold right click. To clear the entire screen, click the clear button. After drawing a digit, click the predict button. A prediction will be displayed along with the confidence percentage.

## Development

The `tox` library is used to run all tests and code formatting. This is automatically installed with the dev requirements. The available options are as follows.

* Run linting checks using `flake8`.

    ```bash
    tox -e lint
    ```

* Run type checks using `mypy`.

    ```bash
    tox -e type
    ```

* Run unit tests `pytest`.

    ```bash
    tox -e test
    ```

* Run all three of the tests above.

    ```bash
    tox
    ```

* Format the code using `black` and `isort` to comply with linting conventions.

    ```bash
    tox -e format
    ```

Upon pull request, merge, or push to the `master` branch, the three tests with `tox` will be run using GitHub Actions. The workflow will fail if any of the tests fail. See `.github/workflows/python-package.yml` for more information on how the CI works.
