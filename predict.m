function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Theta1 size is 25   401
%Theta2 size is 10   26
%5000X401 401X25 -> 500X25 ->5000X26 -> 5000X26 X 26X10 -> 5000 X 10

H1 = 0;
H1 = sigmoid(X * Theta1');
H2 = 0;
H1 = [ones(m, 1), H1];
H2 = sigmoid(H1 * Theta2')
[M, I] = max(H2, [], 2); %the largest element in the row
% we need the column index of the largest element in the row, this index is
% then assigned to p
p = I;


% =========================================================================


end
