import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.checkpoint import Checkpoint


@pytest.fixture
def init_model_and_optimizer():
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 16)
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            out = F.relu(self.fc1(x))
            return self.fc2(out)
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


# TODO: Make this test better.
class TestCheckpoint:
    def test_checkpoint_save_restore(self, tmp_path, init_model_and_optimizer):
        model, optimizer = init_model_and_optimizer
        checkpoint = Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = tmp_path / "ckpts"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test.ckpt"
        checkpoint.save(checkpoint_path)
        assert checkpoint.restore(checkpoint_path)
