import pytest
import os
import pandas as pd
from app.fl.data import FLDataHandler

@pytest.fixture
def dummy_csv_path(tmpdir):
    path = os.path.join(tmpdir, "dummy_data.csv")
    
    # 7 features + 1 target
    columns = ['age_group', 'sex', 'bmi', 'smoked_100_cigarettes', 
               'diabetes_diagnosis', 'heart_attack_history', 'stroke_history',
               'heart_disease_diagnosis']
    data = [
        ['1.0', '1.0', 2500, '1.0', '1.0', '1.0', '1.0', '1.0'],
        ['2.0', '2.0', 3000, '2.0', '2.0', '2.0', '2.0', '2.0']
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)
    return path

def test_fldatahandler_load_data(dummy_csv_path):
    handler = FLDataHandler()
    X, y = handler.load_data(dummy_csv_path)
    
    assert X.shape == (2, 23)
    assert y.shape == (2,)
    assert list(y) == [1, 0]
