import pytest
import os
import torch
from app.fl.model import FLModel

def test_flmodel_initialization():
    model = FLModel(input_size=23)
    assert model is not None
    assert isinstance(model.model, torch.nn.Module)

def test_flmodel_get_set_weights():
    model1 = FLModel(input_size=23)
    model2 = FLModel(input_size=23)
    
    weights1 = model1.get_weights()
    model2.set_weights(weights1)
    
    weights2 = model2.get_weights()
    
    for k in weights1.keys():
        assert torch.equal(weights1[k], weights2[k])

def test_flmodel_forward_pass():
    model = FLModel(input_size=23)
    dummy_input = torch.randn(10, 23)
    
    # predict_proba expects numpy or list, outputs numpy
    probs = model.predict_proba(dummy_input.numpy())
    assert probs.shape == (10, 2)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_flmodel_save_load(tmpdir):
    model1 = FLModel()
    filepath = os.path.join(tmpdir, "model.pkl")
    
    model1.save(filepath)
    assert os.path.exists(filepath)
    
    model2 = FLModel()
    success = model2.load(filepath)
    assert success is True
