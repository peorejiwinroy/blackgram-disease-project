import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
import numpy as np

# Quantum Config
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_net(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridBlackgramNet(nn.Module):
    def __init__(self, num_classes=5, use_quantum=True):
        super(HybridBlackgramNet, self).__init__()
        self.use_quantum = use_quantum
        
        # Base: ResNet50
        self.resnet = models.resnet50(weights='DEFAULT')
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove default head

        if self.use_quantum:
            self.pre_q = nn.Linear(num_ftrs, n_qubits)
            self.q_layer = qml.qnn.TorchLayer(quantum_net, {"weights": (3, n_qubits)})
            self.post_q = nn.Linear(n_qubits, num_classes)
        else:
            self.standard_fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        if self.use_quantum:
            x = torch.sigmoid(self.pre_q(x)) * np.pi
            x = self.q_layer(x)
            x = self.post_q(x)
        else:
            x = self.standard_fc(x)
        return x
