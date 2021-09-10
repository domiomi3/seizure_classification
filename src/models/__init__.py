from .base_model import BaseModel
from .cnn_dense import CNNDenseModel
from .cnn_lstm import CNNLSTMModel


ModelMapping = {
    "cnn_dense": CNNDenseModel,
    "cnn_lstm": CNNLSTMModel,
    "gat_lstm": 'jeszcze nie'
}
