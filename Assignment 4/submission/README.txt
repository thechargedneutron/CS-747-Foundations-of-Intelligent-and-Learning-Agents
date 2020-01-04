Instructions for Running the Code
---------------------------------

1. For part A : `./simulate.sh A`
2. For part B : `./simulate.sh B`
3. For part C : `./simulate.sh C`

Please note that this is case sensitive. You may also need to execute `chmod +x simulate.sh`


Modifying Board Dimensions
--------------------------
Please note that board dimensions have been saved in a separate text file named `board.txt`. The format of the file is as follows

number of rows
number of columns
start state
end state
wind direction (number of elements should be equal to number of columns)


Modifying sarsa(0) agent parameters
-----------------------------------
For simplicity of running the code, sarsa arguments are not passed using command line but are set in the file `simulate.sh`. The following parameters can be modified

alpha - defaults to 0.5
gamma - defaults to 1.0
epsilon - defaults to 0.1
num_steps - defaults to 8000

Code Implementations
--------------------
1. Implementation for sarsa agent can be seen in the file `main.py` with the name of the function as evaluate_sarsa_0
2. Implementation for the board and the environment can be found in the file `main.py` with the name of the function as next_state_decoder