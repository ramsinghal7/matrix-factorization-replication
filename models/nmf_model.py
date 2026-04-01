import numpy as np

class NMF:
    """
    Non-negative Matrix Factorization
    
    Key difference from PMF: All factors must be >= 0
    This makes factors more interpretable and improves novelty
    """
    
    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01, 
                 regularization=0.01, n_epochs=25):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        
        # Initialize with POSITIVE random values (key for NMF!)
        # Use uniform distribution in [0, 1]
        self.user_factors = np.abs(np.random.uniform(0, 0.1, (n_users, n_factors)))
        self.item_factors = np.abs(np.random.uniform(0, 0.1, (n_items, n_factors)))
        
        # Biases (can be negative)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0
    
    def predict(self, user_id, item_id):
        """Predict rating"""
        prediction = self.global_bias
        prediction += self.user_bias[user_id]
        prediction += self.item_bias[item_id]
        prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return prediction
    
    def train(self, train_data):
        """Train with gradient descent + non-negativity constraint"""
        # Calculate global bias
        self.global_bias = train_data['rating'].mean()
        
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = row['rating']
                
                # Prediction and error
                pred = self.predict(user_id, item_id)
                error = rating - pred
                
                # Update biases (same as BiasedMF)
                self.user_bias[user_id] += self.learning_rate * (error - self.reg * self.user_bias[user_id])
                self.item_bias[item_id] += self.learning_rate * (error - self.reg * self.item_bias[item_id])
                
                # Update factors with gradient descent
                user_factor_update = error * self.item_factors[item_id] - self.reg * self.user_factors[user_id]
                item_factor_update = error * self.user_factors[user_id] - self.reg * self.item_factors[item_id]
                
                self.user_factors[user_id] += self.learning_rate * user_factor_update
                self.item_factors[item_id] += self.learning_rate * item_factor_update
                
                # KEY DIFFERENCE: Enforce non-negativity constraint
                # Set any negative values to 0
                self.user_factors[user_id] = np.maximum(0, self.user_factors[user_id])
                self.item_factors[item_id] = np.maximum(0, self.item_factors[item_id])
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")