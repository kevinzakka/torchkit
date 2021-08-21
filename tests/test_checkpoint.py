import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.checkpoint import Checkpoint, CheckpointManager


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


class TestCheckpoint:
    def test_checkpoint_save_restore(self, tmp_path, init_model_and_optimizer):
        model, optimizer = init_model_and_optimizer
        checkpoint = Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = tmp_path / "ckpts"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test.ckpt"
        checkpoint.save(checkpoint_path)
        assert checkpoint.restore(checkpoint_path)

    def test_checkpoint_save_partial_restore(self, tmp_path, init_model_and_optimizer):
        model, optimizer = init_model_and_optimizer
        checkpoint = Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = tmp_path / "ckpts"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test.ckpt"
        checkpoint.save(checkpoint_path)
        assert Checkpoint(model=model).restore(checkpoint_path)
        assert Checkpoint(optimizer=optimizer).restore(checkpoint_path)

    def test_checkpoint_save_faulty_restore(self, tmp_path, init_model_and_optimizer):
        model, optimizer = init_model_and_optimizer
        checkpoint = Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = tmp_path / "ckpts"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test.ckpt"
        checkpoint.save(checkpoint_path)
        model.fc2 = nn.Linear(2, 2)  # Purposely modify model.
        assert not checkpoint.restore(checkpoint_path)

    def test_checkpoint_manager(self, tmp_path, init_model_and_optimizer):
        model, optimizer = init_model_and_optimizer
        ckpt_dir = tmp_path / "ckpts"
        checkpoint_manager = CheckpointManager(
            ckpt_dir,
            max_to_keep=5,
            model=model,
            optimizer=optimizer,
        )
        global_step = checkpoint_manager.restore_or_initialize()
        assert global_step == 0
        for i in range(10):
            checkpoint_manager.save(i)
        available_ckpts = checkpoint_manager.list_checkpoints(ckpt_dir)
        assert len(available_ckpts) == 5
        ckpts = [int(d.stem) for d in available_ckpts]
        expected = list(range(5, 10))
        assert all([a == b for a, b in zip(ckpts, expected)])
        global_step = checkpoint_manager.restore_or_initialize()
        assert global_step == 9
        assert int(checkpoint_manager.latest_checkpoint.stem) == 9
