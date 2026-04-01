import numpy as np
import pandas as pd
from collections import defaultdict

class RecommenderEvaluator:
    def __init__(self, model, test_data, train_data, n_users, n_items, threshold=4.0):
        """
        model: trained MF model
        test_data: test set DataFrame
        train_data: train set DataFrame
        n_users: total number of users
        n_items: total number of items
        threshold: rating threshold to consider an item as "relevant"
        """
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.threshold = threshold
        
        # Pre-compute user-item mappings for efficiency
        self._build_mappings()
        
    def _build_mappings(self):
        """Build dictionaries for faster lookup"""
        # Items rated in training set per user
        self.train_items = defaultdict(set)
        for _, row in self.train_data.iterrows():
            self.train_items[row['user_id']].add(row['item_id'])
        
        # Test ratings per user
        self.test_ratings = defaultdict(dict)
        for _, row in self.test_data.iterrows():
            self.test_ratings[row['user_id']][row['item_id']] = row['rating']
        
        # Relevant items per user (test items with rating >= threshold)
        self.relevant_items = defaultdict(set)
        for _, row in self.test_data.iterrows():
            if row['rating'] >= self.threshold:
                self.relevant_items[row['user_id']].add(row['item_id'])
    
    def get_user_recommendations(self, user_id, n_items):
        """Get top-N recommendations for a user"""
        # Only predict for items in test set that user hasn't rated in train
        candidate_items = []
        rated_in_train = self.train_items[user_id]
        
        # Get all items from test set for this user
        test_items = set(self.test_ratings[user_id].keys())
        
        # Predict ratings for test items not in training
        predictions = []
        for item_id in test_items:
            if item_id not in rated_in_train:
                pred_rating = self.model.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating (descending) and get top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = [item_id for item_id, _ in predictions[:n_items]]
        
        return top_n
    
    def precision_at_k(self, n_recommendations=10):
        """
        Precision@K: Of the N recommended items, how many were relevant?
        """
        precisions = []
        
        # Only evaluate users who have test ratings
        users_with_test = set(self.test_ratings.keys())
        
        for user_id in users_with_test:
            # Skip users with no relevant items
            if len(self.relevant_items[user_id]) == 0:
                continue
            
            # Get recommendations
            recommended = self.get_user_recommendations(user_id, n_recommendations)
            
            if len(recommended) > 0:
                # How many recommended items are relevant?
                hits = len(set(recommended) & self.relevant_items[user_id])
                precision = hits / len(recommended)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self, n_recommendations=10):
        """
        Recall@K: Of all relevant items, how many did we recommend?
        """
        recalls = []
        
        users_with_test = set(self.test_ratings.keys())
        
        for user_id in users_with_test:
            relevant = self.relevant_items[user_id]
            
            # Skip users with no relevant items
            if len(relevant) == 0:
                continue
            
            # Get recommendations
            recommended = self.get_user_recommendations(user_id, n_recommendations)
            
            # How many relevant items did we recommend?
            hits = len(set(recommended) & relevant)
            recall = hits / len(relevant)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def ndcg_at_k(self, n_recommendations=10):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        """
        ndcgs = []
        
        users_with_test = set(self.test_ratings.keys())
        
        for user_id in users_with_test:
            # Get ordered recommendations
            recommended_items = self.get_user_recommendations(user_id, n_recommendations)
            
            if len(recommended_items) == 0:
                continue
            
            # Get actual ratings from test set
            actual_ratings = self.test_ratings[user_id]
            
            # Calculate DCG
            dcg = 0.0
            for idx, item_id in enumerate(recommended_items):
                if item_id in actual_ratings:
                    relevance = actual_ratings[item_id]
                    # Use relevance gain: 2^rel - 1 (standard NDCG formula)
                    gain = (2 ** relevance - 1)
                    discount = np.log2(idx + 2)
                    dcg += gain / discount
            
            # Calculate IDCG (Ideal DCG)
            if len(actual_ratings) > 0:
                ideal_ratings = sorted(actual_ratings.values(), reverse=True)[:n_recommendations]
                idcg = sum((2 ** rel - 1) / np.log2(idx + 2) 
                          for idx, rel in enumerate(ideal_ratings))
                
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0


def calculate_mae(model, test_data):
    """Calculate MAE"""
    errors = []
    for _, row in test_data.iterrows():
        pred = model.predict(int(row['user_id']), int(row['item_id']))
        errors.append(abs(row['rating'] - pred))
    return np.mean(errors)