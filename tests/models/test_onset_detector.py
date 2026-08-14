"""Automated verification suite testing the central OnsetDetector base architecture.

Validates checkpoint serialization paths, automated hardware detection, state tracking
loops, POSIX eviction signal routines, and file housekeeping boundaries safely.
"""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from score2dataset.models.onset_detector import OnsetDetector

# ---1. LIGHTWEIGHT CONCRETE TESTING HARNESS---


class ConcreteTestDetector(OnsetDetector):
    """Subclass implementation of the abstract contract for pipeline validation."""

    def initialize_components(self, device: torch.device) -> None:
        """Mock components builder to assign basic state parameters safely."""
        self._active_model = MagicMock(spec=torch.nn.Module)
        self._active_optimizer = MagicMock(spec=torch.optim.Optimizer)

    def get_dataloader(self) -> DataLoader:
        """DataLoader Fix: Returns a real PyTorch DataLoader to satisfy supertype contract."""
        dataset = TensorDataset(torch.tensor([1.0]), torch.tensor([0.0]))
        return DataLoader(dataset, batch_size=1)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], device: torch.device
    ) -> torch.Tensor:
        """Signature Fix: Reverted parameter name to device to match contract."""
        # Unused variable baseline declaration to satisfy loose linters
        _ = device
        _ = batch
        return torch.tensor(0.42, requires_grad=True)


# ---2. STATE INITIALIZATION AND UTILITY MATRIX TESTS---


def test_get_active_model_guards_against_uninitialized_state():
    """Verifies that calling the model context before initialization throws an explicit error."""
    detector = ConcreteTestDetector()
    with pytest.raises(RuntimeError, match="model network has not been initialized"):
        detector.get_active_model()


@pytest.mark.parametrize(
    "cuda_available, expected_device", [(True, "cuda"), (False, "cpu")]
)
@patch("torch.cuda.is_available")
def test_get_device_resolves_hardware_states_deterministically(
    mock_cuda_check, cuda_available, expected_device
):
    """Verifies that the hardware allocator flags device targets accurately."""
    mock_cuda_check.return_value = cuda_available
    detector = ConcreteTestDetector()

    resolved_device = detector.get_device()
    assert resolved_device.type == expected_device


# ---3. SERIALIZATION AND DESERIALIZATION MATRIX TESTS---


def test_checkpoint_saving_and_loading_lifecycle(tmp_path):
    """Verifies that state dictionaries, epochs, and parameters restore seamlessly via public APIs."""
    detector = ConcreteTestDetector()
    detector.initialize_components(torch.device("cpu"))
    detector.set_active_epoch(14)

    # TYPE FIX: Cast objects to MagicMock explicitly to bypass MethodType complaints
    active_net = cast(MagicMock, detector.get_active_model())
    active_opt = cast(MagicMock, detector.get_active_optimizer())

    assert active_opt is not None
    active_net.state_dict.return_value = {"weight_layer": 0.9}
    active_opt.state_dict.return_value = {"momentum": 0.1}

    checkpoint_file = tmp_path / "outputs" / "checkpoints" / "test_state.pt"

    # 1. Test Save Operations
    detector.save_checkpoint(
        checkpoint_file, layout_info="Test Backup Summary Configuration"
    )
    assert checkpoint_file.exists()

    # 2. Reset the runtime tracker instances to baseline state conditions
    fresh_detector = ConcreteTestDetector()
    fresh_detector.initialize_components(torch.device("cpu"))

    # 3. Test Load Operations and verify layer mode conversions
    resume_epoch = fresh_detector.load_checkpoint(checkpoint_file, torch.device("cpu"))

    assert resume_epoch == 15
    assert fresh_detector.get_active_epoch() == 15

    fresh_net = cast(MagicMock, fresh_detector.get_active_model())
    fresh_opt = cast(MagicMock, fresh_detector.get_active_optimizer())

    assert fresh_net.load_state_dict.called
    assert fresh_net.eval.called
    assert fresh_opt is not None and fresh_opt.load_state_dict.called


# ---4. SYSTEM EVICTION AND SIGNAL RECOVERY TESTS---


@patch("score2dataset.models.onset_detector.Path.parent")
def test_handle_eviction_serializes_state_and_triggers_clean_exit(
    mock_path_parent, _tmp_path
):
    """Verifies that OS eviction triggers capture metrics and raise a clean system exit status."""
    detector = ConcreteTestDetector()

    # PROTECTED FIX: Completely removed the direct _active_checkpoint_name assignment line
    # Isolated path parent allocations smoothly
    mock_parent_dir = MagicMock()
    mock_path_parent.return_value = mock_parent_dir

    with (
        patch("sys.exit") as mock_exit,
        patch.object(detector, "save_checkpoint") as mock_save,
    ):
        detector.handle_eviction(None, None)

        assert mock_save.called
        passed_path = mock_save.call_args
        assert "model_emergency_eviction.pt" in str(passed_path)
        mock_exit.assert_called_once_with(0)


# ---5. UNIFORM RUNTIME LIFECYCLE ENGINE TESTS---


@patch("pathlib.Path.unlink")
@patch("pathlib.Path.exists")
def test_complete_train_lifecycle_engine_execution_flow(mock_exists, mock_unlink):
    """Verifies that the core orchestration engine executes sequential optimization loops flawlessly."""
    detector = ConcreteTestDetector()

    with (
        patch.object(detector, "load_checkpoint", return_value=1) as mock_load,
        patch.object(detector, "save_checkpoint") as mock_save,
        patch("signal.signal") as mock_signal,
    ):

        mock_exists.return_value = True

        detector.train(
            total_epochs=2, checkpoint_name="lifecycle_test", checkpoint_interval=1
        )

        # TYPE FIX: Cast verified model references to track the inner framework assertions
        active_net_mock = cast(MagicMock, detector.get_active_model())

        assert mock_signal.called
        assert mock_load.called
        assert active_net_mock.train.called
        assert detector.get_active_epoch() == 2

        assert mock_save.call_count == 2
        assert mock_unlink.called
