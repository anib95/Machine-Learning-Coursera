function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    no_of_features = size(X,2);
    temp = theta;
        for feature=1:no_of_features
            temp(feature) = theta(feature) - 1/m*alpha*sum(((sigmoid(X*theta)) - y).*X(:,feature));	
        end
    theta = temp;
    J_history(iter) = computeCost(X,y,theta); 
end

end
