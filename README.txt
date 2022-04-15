This zip file contains the necessary folders to run the CN2 algorithm.

The zip is structured as follows:
    - documentation     [The folder contains a PDF report with the pseudocode of this implementation, a discussion over the results and how to execute the code]
    - data              [The folder contains three datasets of different sizes used to test the CN2 algorithm]
    - source            [The folder contains the implementation of the CN2 algorithm in Python and an auxiliary file to load the data]
    - runner.py           [An executable python file to test the CN2 algorithm]
    - README.txt        [A README explaining the contents of the zip file]

Running the test
To run the test you need to create a python virtual environment and install in it the following dependencies:
    pandas = "^1.4.1"
    numpy = "^1.22.3"
    scipy = "^1.8.0"
    sklearn = "^0.0"

After the installation you can run the test with the runner.py script.
!!!It is necessary that the runner.py is in the same location that the source and data folders!!!

usage: runner.py [-h] [--short] [--medium] [--long] [--iterations ITERATIONS] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --iterations ITERATIONS, -i ITERATIONS
                        number of times the algorithm is executed (default 5)
  --seed SEED           seed for reproducible results

Datasets:
  By default all datasets are used

  --short, -s           run CN2 with short dataset
  --medium, -m          run CN2 with medium dataset
  --long, -l            run CN2 with long dataset
