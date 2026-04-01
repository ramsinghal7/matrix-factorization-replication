import numpy as np
from collections import defaultdict
import math

class BeyondAccuracyEvaluator:
    def __init__(self, model, test_data, train_data, n_users, n_items):
        """
        Evaluates novelty and diversity of recommendations
        """
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        
        # Pre-compute item popularity from training data
        self._compute_item_popularity()
        
        # Build user-item mappings
        self.train_items = defaultdict(set)
        for _, row in self.train_data.iterrows():
            self.train_items[row['user_id']].add(row['item_id'])
        
        self.test_items = defaultdict(set)
        for _, row in self.test_data.iterrows():
            self.test_items[row['user_id']].add(row['item_id'])
    
    def _compute_item_popularity(self):
        """Compute how popular each item is (how many users rated it)"""
        self.item_popularity = defaultdict(int)
        for _, row in self.train_data.iterrows():
            self.item_popularity[row['item_id']] += 1
        
        # Total number of ratings
        self.total_ratings = len(self.train_data)
    
    def get_user_recommendations(self, user_id, n_items):
        """Get top-N recommendations for a user"""
        rated_in_train = self.train_items[user_id]
        test_items = self.test_items[user_id]
        
        predictions = []
        for item_id in test_items:
            if item_id not in rated_in_train:
                pred_rating = self.model.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = [item_id for item_id, _ in predictions[:n_items]]
        
        return top_n
    
    def novelty_at_k(self, n_recommendations=10):
        """
        Novelty: Measures how "unexpected" or "non-obvious" recommendations are
        Based on item popularity - recommending less popular items = higher novelty
        
        Formula: Novelty = -log2(popularity(item) / total_items)
        Higher novelty = recommending less popular (more surprising) items
        """
        novelties = []
        
        users_with_test = set(self.test_items.keys())
        
        for user_id in users_with_test:
            recommended = self.get_user_recommendations(user_id, n_recommendations)
            
            if len(recommended) == 0:
                continue
            
            # Calculate novelty for each recommended item
            user_novelty = 0.0
            for item_id in recommended:
                # Get item popularity (number of users who rated it)
                pop = self.item_popularity.get(item_id, 0)
                
                if pop > 0:
                    # Normalize by total number of users who rated anything
                    prob = pop / self.total_ratings
                    # Novelty is negative log of probability
                    # More popular items (higher prob) = lower novelty
                    novelty = -math.log2(prob) if prob > 0 else 0
                    user_novelty += novelty
            
            # Average novelty across all recommendations
            avg_novelty = user_novelty / len(recommended)
            novelties.append(avg_novelty)
        
        return np.mean(novelties) if novelties else 0.0
    
    def diversity_at_k(self, n_recommendations=10):
        """
        Diversity: Measures variety within the recommendation list
        Uses item-based dissimilarity (how different items are from each other)
        
        We'll use a simple approach: items are diverse if they're not the same
        For a more sophisticated version, you could use item features/genres
        """
        diversities = []
        
        users_with_test = set(self.test_items.keys())
        
        for user_id in users_with_test:
            recommended = self.get_user_recommendations(user_id, n_recommendations)
            
            if len(recommended) <= 1:
                continue
            
            # Calculate pairwise diversity
            # For simplicity: diversity = 1 if items are different, 0 if same
            # In practice, you'd use item features (genre, category, etc.)
            
            total_pairs = 0
            diverse_pairs = 0
            
            for i in range(len(recommended)):
                for j in range(i + 1, len(recommended)):
                    total_pairs += 1
                    # Items are always different (different IDs)
                    # In real scenarios, compute based on item similarity
                    if recommended[i] != recommended[j]:
                        diverse_pairs += 1
            
            if total_pairs > 0:
                diversity = diverse_pairs / total_pairs
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def intra_list_diversity(self, n_recommendations=10):
        """
        Alternative diversity metric: based on item popularity spread
        More diverse list = recommends items with varied popularity levels
        """
        diversities = []
        
        users_with_test = set(self.test_items.keys())
        
        for user_id in users_with_test:
            recommended = self.get_user_recommendations(user_id, n_recommendations)
            
            if len(recommended) <= 1:
                continue
            
            # Get popularity of recommended items
            popularities = [self.item_popularity.get(item_id, 0) 
                          for item_id in recommended]
            
            if len(popularities) > 1:
                # Diversity = standard deviation of popularities
                # Higher std = more varied popularity = more diverse
                diversity = np.std(popularities)
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0