# YUKI-ANN

The code presented here implements a YUKI algorithm training of the feedforward neural network with multiple hidden layers,

First loads the model parameters and equivalent data from text files. Then, it creates a feedforward neural network with four hidden layers of 4, 6, 10, and 3 neurons, respectively.

Next, there are two parts of training the network using two different methods. In the first part, the network is trained with the backpropagation algorithm (default option of the train function) using gradient descent. The input and output data are Model_Data and Model_Parameters, respectively.

In the second part, a modified version of the particle swarm optimization algorithm, called YUKI, is employed to optimize the weights of the neural network. A random initial population of solutions is generated, and a local search area is defined around the current best solution. Then, for each solution, a new position is generated by adding a weighted combination of two vectors. The first vector points towards the historical best position of the solution, while the second vector points towards the current best solution's position. The weights of these two vectors are defined by a constant EXP, which controls the exploration-exploitation trade-off.

After generating new solutions, their weights are updated in the neural network, and the corresponding error is calculated. Then, the best positions and errors for each solution and the global best position and error are updated.

The algorithm continues until either the maximum number of iterations is reached or the error is below a predefined tolerance level. Finally, the trained neural network is used to make predictions on a new set of input data, and the training results are plotted, including the mean squared error and the regression results.

**Deep Neural Network and YUKI Algorithm for Inner Damage Characterization Based on Elastic Boundary Displacement**. *Lecture Notes in Civil Engineering*. 2023. <a href="https://doi.org/10.1007/978-3-031-24041-6_18" target="_blank"> https://doi.org/10.1007/978-3-031-24041-6_18 </a>

**Damage assessment in laminated composite plates using Modal Strain Energy and YUKI-ANN algorithm**. *Composite Structures*. 2023. <a href="https://doi.org/10.1016/j.compstruct.2022.116272" target="_blank"> https://doi.org/10.1016/j.compstruct.2022.116272 </a>