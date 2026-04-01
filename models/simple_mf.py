import numpy as np
from sklearn.model_selection import train_test_split

class SimpleMF:
    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01, n_epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        # Initialize user and item factor matrices randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def train(self, train_data):
        """Train the model using gradient descent"""
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = row['rating']
                
                # Calculate prediction and error
                pred = self.predict(user_id, item_id)
                error = rating - pred
                
                # Update factors using gradient descent
                self.user_factors[user_id] += self.learning_rate * error * self.item_factors[item_id]
                self.item_factors[item_id] += self.learning_rate * error * self.user_factors[user_id]
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")