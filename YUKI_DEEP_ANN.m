% 

%% load the data and setup the network
clc
clear

% Load model parameters and equivalent data
Model_Parameters = load('Parameters.txt');
Model_Data = load('Data.txt');

% Create a feedforward neural network % Define the number of neurons in each hidden layer
net = feedforwardnet([4 6 10 3]);


%% 1 train the feedforward neural network with Gradient Descent 
%to set the input and output sizes  

net = train(net,Model_Data',Model_Parameters');

%% 2 train the feedforward with YUKI
tic;
EXP=0.5;
Population_size = 50; % Number of solutions 
max_iterations = 100; % Maximum number of iterations
error_tolerance = 1e-6; % Error tolerance

% Create the solutions based on the number of weights in this networrk
num_weights = net.numWeightElements; % Get the number of weights in this network
solutions = rand(Population_size, num_weights); % Initialize the solutions


% Train the network using YUKI
[net, solutions, error] = YUKItrain(net, Model_Data', Model_Parameters', solutions, num_weights, EXP, max_iterations, error_tolerance);

toc;

%% Test the network with new input data


load YUKItrainedNet
predictions  = YUKItrainedNet(Model_Data);

% Calculate the R value for the training
R = corrcoef(Model_Parameters(:), predictions(:));
R = R(1,2);

% Plot the training results
figure(1)
semilogy(1:length(error), error, 'LineWidth', 2)
xlabel('Iteration')
ylabel('Mean Squared Error')
title('Training Results' )

% Plot the regression results
figure(2)
plot(Model_Parameters, predictions, 'bo', 'LineWidth', 2)
xlabel('True Model_Parameters')
ylabel('Predicted Outputs')
title(['Regression Results, R = ' num2str(R)])



%% the YUKI training function 
function [net, solutions, error] = YUKItrain(net, Model_Data, Model_Parameters,  solutions, num_weights, EXP, max_iterations, error_tolerance)

Population_size = size(solutions, 1); % Define solutions
historical_best_position = solutions; % Initialbest solution positions for each solution
solution_best_error = inf(Population_size, 1); % Initial best error for each solution
Center_position = solutions(1, :); % Initial global best position
% MeanBest_position=solutions(1, :); % Initial MeanBest position
Center_error = inf; % Initial global best error
error = zeros(max_iterations, 1); % Initial error array
Dist_MeanBest=ones(1,size(solutions,2)); 


% Opimization loop until the maximum number of iterations is reached
for iteration = 1:max_iterations

    % Calculate the boundaries of the local search area
    Local_bnd=[Center_position-Dist_MeanBest;Center_position+Dist_MeanBest];

    % Check for the external boundaries 
    Local_bnd(Local_bnd>1)=1; 
    Local_bnd(Local_bnd<0)=0; 

    % Generate the local population 
    Local_random_positions = rand(Population_size, num_weights).*(Local_bnd(2,:) - Local_bnd(1,:)) + Local_bnd(1,:);

    for i=1:Population_size % calculate new positiones for the solutions
      if  rand<EXP
          Explore=Local_random_positions(i,:)- historical_best_position(i,:);  
          solutions(i,:)=Local_random_positions(i,:)+Explore;
      else
          Eploite=Local_random_positions(i,:)-Center_position;
          solutions(i,:)=Local_random_positions(i,:)+rand*Eploite;
      end
    end 

    % Chech the solutions out of the top boundaries 
    upid=solutions>1;
    upr = rand(1,nnz(upid));
    solutions(upid) = upr;
    
    % Chech the solutions out of the bottom boundaries 
    downid=solutions<0;
    downr = rand(1,nnz(downid));
    solutions(downid) = downr;


    for i=1:Population_size
        net = setx(net, solutions(i, :)); % Update the weights of the network
        outputs = net(Model_Data); % Calculate the outputs of the network
        solution_error = mean((outputs - Model_Parameters).^2, 'all'); % Calculate the error

        % Update the solution's best position and error
        if solution_error < solution_best_error(i)
            historical_best_position(i, :) = solutions(i, :);
            solution_best_error(i) = solution_error;
        end

        % Update the Center best position and error
        if solution_error < Center_error
            Center_position = solutions(i, :);
            Center_error = solution_error;
        end
    end
     
    % Update the MeanBest position
    MeanBest_position=mean(historical_best_position,1);  

    % Update the size of the local search area
    Dist_MeanBest=(Center_position-MeanBest_position);

    error(iteration) = Center_error;% Store the errors
     
    % Display progress
    disp(iteration)
    disp(Center_error)
      
    % Check if the error is below the tolerance
    if Center_error < error_tolerance
        break;
    end
    
end

% Set the best weights in the network
net = setx(net, Center_position);

% Save the network under name "YUKItrainedNet.mat"
YUKItrainedNet = net;
save YUKItrainedNet

end
