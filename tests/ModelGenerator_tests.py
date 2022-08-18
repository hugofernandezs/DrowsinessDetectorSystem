import pytest
from ModelGenerator import *

def test_create_datasets() -> None:
  trainingDataset, validationDataset = create_datasets("datasets/train")
  assert(trainingDataset != None)
  assert(validationDataset != None)