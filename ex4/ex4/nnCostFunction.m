function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% cost function via forward propagation
X = [ones(m,1) X]; 
a = sigmoid(X*Theta1'); 
a = [ones(m,1) a];
h = sigmoid(a*Theta2');
log1 = log(h);
log2 = log(1-h);
yi = zeros(m,num_labels); % 5000 by 10
for i = 1:num_labels
  yi(:,i) = y == i;
end
J = -1/m *sum(sum(((yi.*log1) + ((1 - yi).*log2))));

% regularization for cost function
theta1 = Theta1(:,2:input_layer_size+1);
theta2 = Theta2(:,2:hidden_layer_size+1);
regular = lambda/(2*m) * (sum(sum(theta1.^2)) + sum(sum(theta2.^2)));
J = J + regular;

% gradient via backpropagation
for t = 1:m
  xt = X(t,:);% 1 by 401
  z1 = xt*Theta1'; % 1 by 25
  a1 = sigmoid(z1); % 1 by 25
  z1 = [ones(1,1) z1]; % 1 by 26
  a1 = [ones(1,1) a1]; % 1 by 26
  h = sigmoid(a1*Theta2'); % 1 by 10
  delta3 = h - yi(t,:); % 1 by 10
  delta2 = (Theta2'*delta3'.*(sigmoidGradient(z1))'); 
  delta2 = delta2(2:end); % 25 by 1
  Theta1_grad = Theta1_grad + delta2*xt; % 25 by 401
  Theta2_grad = Theta2_grad + delta3'*a1; % 10 by 26
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

%regularization for gradient
regular1 = lambda/m * (Theta1(:,2:end)); % 25 by 400
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + regular1; % 25 by 400
regular2 = lambda/m * (Theta2(:,2:end)); % 10 by 25
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + regular2; % 10 by 25


  


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
