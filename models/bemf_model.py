import numpy as np
from scipy.special import expit as sigmoid  # logistic sigmoid function

class BeMF:
    """
    Bernoulli Matrix Factorization
    
    Key features:
    - Uses binary classification for each possible rating
    - Aggregates results to get final prediction + reliability
    - Based on Bernoulli distribution
    """
    
    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01, 
                 regularization=0.01, n_epochs=25):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        
        # Rating scores (for 1-5 scale)
        self.scores = [1, 2, 3, 4, 5]
        self.n_scores = len(self.scores)
        
        # Create separate factor matrices for each score
        # Each score gets its own P and Q matrices
        self.user_factors = {}
        self.item_factors = {}
        
        for score in self.scores:
            self.user_factors[score] = np.random.normal(0, 0.1, (n_users, n_factors))
            self.item_factors[score] = np.random.normal(0, 0.1, (n_items, n_factors))
    
    def predict_score_probability(self, user_id, item_id, score):
        """
        Predict probability that user would give this score to item
        Uses sigmoid to map to [0, 1]
        """
        dot_product = np.dot(self.user_factors[score][user_id], 
                            self.item_factors[score][item_id])
        probability = sigmoid(dot_product)
        return probability
    
    def predict(self, user_id, item_id):
        """
        Predict rating by aggregating probabilities across all scores
        Returns the score with highest probability
        """
        probabilities = []
        
        for score in self.scores:
            prob = self.predict_score_probability(user_id, item_id, score)
            probabilities.append(prob)
        
        # Normalize probabilities to sum to 1
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / self.n_scores] * self.n_scores
        
        # Weighted average of scores by their probabilities
        predicted_rating = sum(score * prob for score, prob in zip(self.scores, probabilities))
        
        return predicted_rating
    
    def train(self, train_data):
        """
        Train using gradient descent with Bernoulli log-likelihood
        """
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = int(row['rating'])
                
                # For each possible score, update factors
                for score in self.scores:
                    # Binary target: 1 if rating == score, 0 otherwise
                    target = 1.0 if rating == score else 0.0
                    
                    # Predicted probability
                    prob = self.predict_score_probability(user_id, item_id, score)
                    
                    # Error (difference from target)
                    error = target - prob
                    
                    # Gradient descent update
                    # Update user factors
                    user_gradient = (error * self.item_factors[score][item_id] - 
                                   self.reg * self.user_factors[score][user_id])
                    self.user_factors[score][user_id] += self.learning_rate * user_gradient
                    
                    # Update item factors
                    item_gradient = (error * self.user_factors[score][user_id] - 
                                   self.reg * self.item_factors[score][item_id])
                    self.item_factors[score][item_id] += self.learning_rate * item_gradient
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")
    
    def predict_with_reliability(self, user_id, item_id):
        """
        Predict rating AND reliability score
        Reliability = probability of the predicted rating
        """
        probabilities = []
        
        for score in self.scores:
            prob = self.predict_score_probability(user_id, item_id, score)
            probabilities.append(prob)
        
        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / self.n_scores] * self.n_scores
        
        # Predicted rating
        predicted_rating = sum(score * prob for score, prob in zip(self.scores, probabilities))
        
        # Reliability = max probability (confidence in prediction)
        reliability = max(probabilities)
        
        return predicted_rating, reliability