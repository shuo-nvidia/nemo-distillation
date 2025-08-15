# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile
from pathlib import Path

import pytest

from nemo_rl.utils.config import load_config


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_config(config_dir: Path, name: str, content: str):
    """Create a test config file."""
    config_path = config_dir / name
    config_path.write_text(content)
    return config_path


def test_single_inheritance(temp_config_dir):
    """Test basic inheritance from a single parent."""
    # Create parent config
    parent_content = """
    common:
      value: 42
    parent_only:
      value: 100
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    common:
      value: 43
    child_only:
      value: 200
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 43  # Child overrides parent
    assert config.parent_only.value == 100  # Parent value preserved
    assert config.child_only.value == 200  # Child-only value exists


def test_multiple_inheritance(temp_config_dir):
    """Test inheritance from multiple parents."""
    # Create first parent
    parent1_content = """
    common:
      value: 42
    parent1_only:
      value: 100
    """
    create_test_config(temp_config_dir, "parent1.yaml", parent1_content)

    # Create second parent
    parent2_content = """
    common:
      value: 43
    parent2_only:
      value: 200
    """
    create_test_config(temp_config_dir, "parent2.yaml", parent2_content)

    # Create child config
    child_content = """
    defaults:
      - parent1.yaml
      - parent2.yaml
    common:
      value: 44
    child_only:
      value: 300
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 44  # Child overrides both parents
    assert config.parent1_only.value == 100  # First parent value preserved
    assert config.parent2_only.value == 200  # Second parent value preserved
    assert config.child_only.value == 300  # Child-only value exists


def test_absolute_path_inheritance(temp_config_dir):
    """Test inheritance using absolute paths."""
    # Create parent config
    parent_content = """
    common:
      value: 42
    """
    parent_path = create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config with absolute path
    child_content = f"""
    defaults: {parent_path}
    common:
      value: 43
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 43  # Child overrides parent


def test_no_inheritance(temp_config_dir):
    """Test config without inheritance."""
    content = """
    common:
      value: 42
    """
    config_path = create_test_config(temp_config_dir, "config.yaml", content)

    # Load and verify
    config = load_config(config_path)
    assert config.common.value == 42


def test_nested_inheritance(temp_config_dir):
    """Test nested inheritance (parent inherits from grandparent)."""
    # Create grandparent config
    grandparent_content = """
    common:
      value: 42
    grandparent_only:
      value: 100
    """
    create_test_config(temp_config_dir, "grandparent.yaml", grandparent_content)

    # Create parent config
    parent_content = """
    defaults: grandparent.yaml
    common:
      value: 43
    parent_only:
      value: 200
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    common:
      value: 44
    child_only:
      value: 300
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 44  # Child overrides all
    assert config.grandparent_only.value == 100  # Grandparent value preserved
    assert config.parent_only.value == 200  # Parent value preserved
    assert config.child_only.value == 300  # Child-only value exists


def test_interpolation(temp_config_dir):
    """Test that interpolation works with inherited configs."""
    # Create parent config
    parent_content = """
    base_value: 42
    derived:
      value: ${base_value}
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    base_value: 43
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.base_value == 43
    assert config.derived.value == 43  # Interpolation uses child's base_value


def test_parse_hydra_overrides():
    """Test parsing and applying Hydra overrides."""
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import OverridesError, parse_hydra_overrides

    # Create initial config
    cfg = OmegaConf.create(
        {
            "model": {"type": "default", "hidden_size": 768},
            "training": {"batch_size": 32, "learning_rate": 1e-4},
        }
    )

    # Test basic override
    overrides = ["model.type=transformer"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.type == "transformer"
    assert updated_cfg.model.hidden_size == 768  # Unchanged

    # Test nested override
    overrides = ["model.hidden_size=1024"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.hidden_size == 1024

    # Test multiple overrides
    overrides = ["training.batch_size=64", "training.learning_rate=2e-4"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.training.batch_size == 64
    assert updated_cfg.training.learning_rate == 2e-4

    # Test invalid override
    overrides = ["nonexistent.key=value"]
    with pytest.raises(OverridesError):
        parse_hydra_overrides(cfg, overrides)

    # Test invalid syntax
    overrides = ["invalid.syntax"]
    with pytest.raises(OverridesError):
        parse_hydra_overrides(cfg, overrides)

    # Test empty overrides
    overrides = []
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg == cfg  # Config should be unchanged

    # Test override additions and deletions
    overrides = [
        "+model.num_layers=12",
        "++model.type=transformer",
        "~training.batch_size",
    ]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.num_layers == 12
    assert updated_cfg.model.type == "transformer"
    assert "batch_size" not in updated_cfg.training
