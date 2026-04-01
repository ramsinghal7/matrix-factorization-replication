import numpy as np

class URP:
    """
    User Ratings Profile Model
    Generative latent variable model based on user rating patterns
    """
    
    def __init__(self, n_users, n_items, n_factors=10, n_epochs=25, alpha=0.5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.alpha = alpha  # Dirichlet prior
        
        # Rating scores (1-5)
        self.scores = [1, 2, 3, 4, 5]
        self.n_scores = len(self.scores)
        
        # User topic distributions (theta)
        # Each user has a distribution over K topics
        self.theta = np.random.dirichlet([alpha] * n_factors, n_users)
        
        # Topic-word distributions (beta)
        # For each topic and item, distribution over ratings
        self.beta = np.random.dirichlet([1.0] * self.n_scores, (n_factors, n_items))
        
        # Topic assignments (z)
        self.z = {}
        
        # Global bias
        self.global_bias = 3.0
    
    def predict(self, user_id, item_id):
        """
        Predict rating using expected value over topics
        """
        prediction = 0.0
        
        # For each topic
        for k in range(self.n_factors):
            # User's preference for topic k
            topic_weight = self.theta[user_id, k]
            
            # Expected rating for this item under topic k
            rating_dist = self.beta[k, item_id]
            expected_rating = sum(score * prob for score, prob in zip(self.scores, rating_dist))
            
            # Add weighted contribution
            prediction += topic_weight * expected_rating
        
        return np.clip(prediction, 1, 5)
    
    def train(self, train_data):
        """
        Train using simplified variational inference
        """
        # Calculate global bias
        self.global_bias = train_data['rating'].mean()
        
        # Build user-item rating lookup
        ratings_dict = {}
        for _, row in train_data.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            rating = int(row['rating'])
            ratings_dict[(uid, iid)] = rating
        
        # Build list of ratings per user
        user_items = {u: [] for u in range(self.n_users)}
        for (u, i), r in ratings_dict.items():
            user_items[u].append((i, r))
        
        print(f"Training URP on {len(train_data):,} ratings...")
        
        for epoch in range(self.n_epochs):
            # Update topic assignments (z) and counts
            topic_counts = np.zeros((self.n_users, self.n_factors))
            topic_item_counts = np.zeros((self.n_factors, self.n_items, self.n_scores))
            
            # E-step: Assign topics to ratings
            for u in range(self.n_users):
                if len(user_items[u]) == 0:
                    continue
                
                for item_id, rating in user_items[u]:
                    rating_idx = rating - 1  # Convert to 0-indexed
                    
                    # Compute probability of each topic
                    topic_probs = np.zeros(self.n_factors)
                    for k in range(self.n_factors):
                        topic_probs[k] = (self.theta[u, k] * 
                                        self.beta[k, item_id, rating_idx])
                    
                    # Normalize
                    if topic_probs.sum() > 0:
                        topic_probs /= topic_probs.sum()
                    else:
                        topic_probs = np.ones(self.n_factors) / self.n_factors
                    
                    # Update counts (soft assignment)
                    for k in range(self.n_factors):
                        topic_counts[u, k] += topic_probs[k]
                        topic_item_counts[k, item_id, rating_idx] += topic_probs[k]
            
            # M-step: Update parameters
            # Update theta (user topic distributions)
            for u in range(self.n_users):
                self.theta[u] = (topic_counts[u] + self.alpha) / (
                    topic_counts[u].sum() + self.n_factors * self.alpha
                )
            
            # Update beta (topic-item-rating distributions)
            for k in range(self.n_factors):
                for i in range(self.n_items):
                    counts = topic_item_counts[k, i] + 1.0
                    self.beta[k, i] = counts / counts.sum()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")
        
        print("URP training completed!")