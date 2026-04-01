import numpy as np

class BiasedMF:
    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01, 
                 regularization=0.01, n_epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = regularization  # Regularization to prevent overfitting
        self.n_epochs = n_epochs
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0
    
    def predict(self, user_id, item_id):
        """Predict rating with biases"""
        prediction = self.global_bias
        prediction += self.user_bias[user_id]
        prediction += self.item_bias[item_id]
        prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return prediction
    
    def train(self, train_data):
        """Train with gradient descent"""
        # Calculate global bias (mean of all ratings)
        self.global_bias = train_data['rating'].mean()
        
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = row['rating']
                
                # Prediction and error
                pred = self.predict(user_id, item_id)
                error = rating - pred
                
                # Update biases with regularization
                self.user_bias[user_id] += self.learning_rate * (error - self.reg * self.user_bias[user_id])
                self.item_bias[item_id] += self.learning_rate * (error - self.reg * self.item_bias[item_id])
                
                # Update factors with regularization
                user_factor_update = error * self.item_factors[item_id] - self.reg * self.user_factors[user_id]
                item_factor_update = error * self.user_factors[user_id] - self.reg * self.item_factors[item_id]
                
                self.user_factors[user_id] += self.learning_rate * user_factor_update
                self.item_factors[item_id] += self.learning_rate * item_factor_update
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")