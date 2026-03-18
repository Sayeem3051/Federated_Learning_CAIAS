import pytest
import os
import pandas as pd
import torch
from app.fl.aggregator import FLAggregator

def test_aggregator_initialization():
    agg = FLAggregator()
    assert agg.round == 0
    assert len(agg.client_weights) == 0

def test_aggregator_add_client_update():
    agg = FLAggregator()
    
    # Mock weights
    mock_weights = agg.global_model.get_weights()
    
    agg.add_client_update(mock_weights, n_samples=100, metrics={'accuracy': 0.85, 'loss': 0.5})
    
    assert len(agg.client_weights) == 1
    assert agg.client_sizes[0] == 100
    assert agg.client_metrics[0]['accuracy'] == 0.85

def test_aggregator_aggregate():
    agg = FLAggregator()
    
    # Add two dummy updates
    weights1 = agg.global_model.get_weights()
    weights2 = agg.global_model.get_weights()
    
    agg.add_client_update(weights1, n_samples=100, metrics={'accuracy': 0.8, 'loss': 0.6})
    agg.add_client_update(weights2, n_samples=100, metrics={'accuracy': 0.9, 'loss': 0.4})
    
    success = agg.aggregate()
    
    assert success is True
    assert agg.round == 1
    assert len(agg.client_weights) == 0
    assert len(agg.history['rounds']) == 1
    assert agg.history['accuracy'][0] == 0.85  # average of 0.8 and 0.9
