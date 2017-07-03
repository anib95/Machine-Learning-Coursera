function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);
temp_vector = p;
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

no_of_features = size(X,2);
temp_matrix = zeros(m,no_of_features);
for iter = 1:no_of_features
    temp_vector = theta(iter) * X(:,iter);
    temp_matrix(:,iter) = temp_vector; 
end
p = sum(temp_matrix,2);
positive = find(p>=0);
negative = find(p<0);
for i=1:size(positive)
    p(positive(i)) = 1;
end
for i = 1:size(negative)
    p(negative(i)) = 0;
end
% =========================================================================


end
