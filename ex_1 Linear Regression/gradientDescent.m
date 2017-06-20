function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    actual_data = X(:,2);
    hypothesis = theta(1) + (theta(2)*actual_data);
    delta1 = 1/m * alpha * sum(hypothesis - y);
    temp1 = theta(1) - delta1;
    delta2 = 1/m * alpha * sum((hypothesis - y).*actual_data);
    temp2 = theta(2) - delta2;
    theta = [temp1;temp2];
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
