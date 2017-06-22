function J = computeCost(X,y,theta)
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
 J = 0;

 no_of_features = size(X,2);
 hypothesis = sigmoid(X * theta);
 J = -(1/m) * sum(y.*log(hypothesis) + (1-y).*log(1-hypothesis));

end
