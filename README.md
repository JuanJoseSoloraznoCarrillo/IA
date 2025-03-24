# Perceptron Training and Evaluation

## Overview

This project demonstrates the training and evaluation of a Perceptron model using a custom dataset. The script `test_perceptron.py` handles data loading, model training, and evaluation, and visualizes the results.

## Description

The script `test_perceptron.py` performs the following tasks:
1. Loads training and testing data.
2. Plots the training and testing data.
3. Initializes and trains a Perceptron model.
4. Plots the training loss over epochs.
5. Evaluates the model on the test data.
6. Plots the predicted results.
7. Calculates and prints the accuracy of the model.
8. Plots the decision boundary of the Perceptron model.

## Dependencies

This project requires the following Python libraries:
- `os`
- `math`
- `numpy`
- `matplotlib` (for plotting, assumed to be used in `misc` module)

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

1. Ensure you have the necessary data and modules ([misc](http://_vscodecontentref_/1) and [perceptron](http://_vscodecontentref_/2)) in the same directory as [test_perceptron.py](http://_vscodecontentref_/3).
2. Run the script:
    ```sh
    python test_perceptron.py
    ```

## Script Details

### Imports

- [os](http://_vscodecontentref_/4), [math](http://_vscodecontentref_/5), [numpy](http://_vscodecontentref_/6): Standard Python libraries.
- [get_data](http://_vscodecontentref_/7), [plot_data](http://_vscodecontentref_/8): Functions from the [misc](http://_vscodecontentref_/9) module for data handling and plotting.
- [Perceptron](http://_vscodecontentref_/10), [FOLDER](http://_vscodecontentref_/11), [training_loop](http://_vscodecontentref_/12), [get_line_boundary](http://_vscodecontentref_/13), [plot_loss](http://_vscodecontentref_/14): Classes and functions from the [perceptron](http://_vscodecontentref_/15) module for model training and evaluation.

### Main Workflow

1. **Data Loading**:
    ```python
    epochs, learning_rate, x_train, y_train, x_test, y_test, verbose = get_data()
    ```

2. **Data Plotting**:
    ```python
    plot_data('train_data', x_train=x_train, y_train=y_train)
    plot_data('test_data', x_test=x_test)
    ```

3. **Model Initialization and Training**:
    ```python
    perceptron = Perceptron(num_inputs=2, learning_rate=learning_rate)
    cost_history = training_loop(perceptron, inputs=x_train, target=y_train, epochs=100)
    ```

4. **Plotting Training Loss**:
    ```python
    plot_loss(epochs=100, cost_history=cost_history)
    ```

5. **Model Evaluation**:
    ```python
    y_pred = [perceptron.predict(x) for x in x_test]
    y_pred = np.array(y_pred)
    print('>> Accuracy: {}'.format(Perceptron.accuracy(y_test, y_pred)))
    plot_data('prediction', x_test, y_pred)
    ```

6. **Plotting Decision Boundary**:
    ```python
    _max = math.ceil(max(x_test.flatten()))
    _min = int(_max / 2)
    get_line_boundary(perceptron, save=True, _min_=_min, _max_=_max)
    ```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact Juan Jose Solorzano at [juanjose.solorzano.c@gmail.com](mailto:juanjose.solorzano.c@gmail.com).