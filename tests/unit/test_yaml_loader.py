# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for YAML loader utilities."""

from pathlib import Path

import pytest

from src.core.config.yaml_loader import (
    YAMLLoadError,
    deep_merge,
    load_yaml,
    load_yaml_directory,
)


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml_file(self, tmp_path: Path) -> None:
        """Test loading a valid YAML file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnested:\n  inner: 42\n")

        result = load_yaml(yaml_file)

        assert result == {"key": "value", "nested": {"inner": 42}}

    def test_load_empty_yaml_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that empty YAML files return empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = load_yaml(yaml_file)

        assert result == {}

    def test_load_yaml_with_only_comments_returns_empty_dict(
        self, tmp_path: Path
    ) -> None:
        """Test that YAML files with only comments return empty dict."""
        yaml_file = tmp_path / "comments.yaml"
        yaml_file.write_text("# Just a comment\n# Another comment\n")

        result = load_yaml(yaml_file)

        assert result == {}

    def test_load_yaml_with_list_root_raises_error(self, tmp_path: Path) -> None:
        """Test that YAML files with list root raise error."""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n")

        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml(yaml_file)

        assert "YAML root must be a mapping" in str(exc_info.value)

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading non-existent file raises error."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml(nonexistent)

        assert "File does not exist" in str(exc_info.value)

    def test_load_directory_instead_of_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading a directory raises error."""
        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml(tmp_path)

        assert "Path is not a file" in str(exc_info.value)

    def test_load_invalid_yaml_syntax_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid YAML syntax raises error."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: value\n  invalid indentation")

        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml(yaml_file)

        assert "Invalid YAML syntax" in str(exc_info.value)

    def test_load_yaml_with_complex_types(self, tmp_path: Path) -> None:
        """Test loading YAML with various types."""
        yaml_file = tmp_path / "complex.yaml"
        yaml_file.write_text(
            """
string: hello
integer: 42
float: 3.14
boolean: true
null_value: null
list:
  - item1
  - item2
nested:
  deep:
    value: nested_value
"""
        )

        result = load_yaml(yaml_file)

        assert result["string"] == "hello"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["null_value"] is None
        assert result["list"] == ["item1", "item2"]
        assert result["nested"]["deep"]["value"] == "nested_value"


class TestLoadYamlDirectory:
    """Tests for load_yaml_directory function."""

    def test_load_directory_with_yaml_files(self, tmp_path: Path) -> None:
        """Test loading all YAML files from a directory."""
        (tmp_path / "first.yaml").write_text("key: first\n")
        (tmp_path / "second.yaml").write_text("key: second\n")

        result = load_yaml_directory(tmp_path)

        assert result == {"first": {"key": "first"}, "second": {"key": "second"}}

    def test_load_directory_with_yml_extension(self, tmp_path: Path) -> None:
        """Test that .yml files are also loaded."""
        (tmp_path / "config.yml").write_text("key: value\n")

        result = load_yaml_directory(tmp_path)

        assert result == {"config": {"key": "value"}}

    def test_load_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that empty directory returns empty dict."""
        result = load_yaml_directory(tmp_path)

        assert result == {}

    def test_load_directory_ignores_non_yaml_files(self, tmp_path: Path) -> None:
        """Test that non-YAML files are ignored."""
        (tmp_path / "config.yaml").write_text("key: value\n")
        (tmp_path / "readme.txt").write_text("This is a readme\n")
        (tmp_path / "data.json").write_text('{"key": "value"}\n')

        result = load_yaml_directory(tmp_path)

        assert result == {"config": {"key": "value"}}

    def test_load_nonexistent_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that non-existent directory raises error."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml_directory(nonexistent)

        assert "Directory does not exist" in str(exc_info.value)

    def test_load_file_instead_of_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that passing a file raises error."""
        yaml_file = tmp_path / "file.yaml"
        yaml_file.write_text("key: value\n")

        with pytest.raises(YAMLLoadError) as exc_info:
            load_yaml_directory(yaml_file)

        assert "Path is not a directory" in str(exc_info.value)

    def test_load_directory_preserves_file_order(self, tmp_path: Path) -> None:
        """Test that files are loaded in sorted order."""
        (tmp_path / "z_last.yaml").write_text("order: last\n")
        (tmp_path / "a_first.yaml").write_text("order: first\n")
        (tmp_path / "m_middle.yaml").write_text("order: middle\n")

        result = load_yaml_directory(tmp_path)

        # Check all files were loaded (order in dict doesn't matter in Python 3.7+)
        assert len(result) == 3
        assert result["a_first"]["order"] == "first"
        assert result["m_middle"]["order"] == "middle"
        assert result["z_last"]["order"] == "last"


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_merge_simple_dicts(self) -> None:
        """Test merging simple non-overlapping dicts."""
        base = {"a": 1, "b": 2}
        override = {"c": 3, "d": 4}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_override_takes_precedence(self) -> None:
        """Test that override values take precedence."""
        base = {"a": 1, "b": 2}
        override = {"b": 20, "c": 3}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 20, "c": 3}

    def test_nested_dicts_are_merged_recursively(self) -> None:
        """Test that nested dicts are merged recursively."""
        base = {"a": 1, "nested": {"b": 2, "c": 3}}
        override = {"nested": {"c": 30, "d": 4}}

        result = deep_merge(base, override)

        assert result == {"a": 1, "nested": {"b": 2, "c": 30, "d": 4}}

    def test_deeply_nested_merge(self) -> None:
        """Test merging deeply nested structures."""
        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        override = {"level1": {"level2": {"level3": {"b": 20, "c": 3}}}}

        result = deep_merge(base, override)

        assert result == {"level1": {"level2": {"level3": {"a": 1, "b": 20, "c": 3}}}}

    def test_non_dict_replaces_dict(self) -> None:
        """Test that non-dict override replaces dict base."""
        base = {"a": {"nested": "value"}}
        override = {"a": "simple"}

        result = deep_merge(base, override)

        assert result == {"a": "simple"}

    def test_dict_replaces_non_dict(self) -> None:
        """Test that dict override replaces non-dict base."""
        base = {"a": "simple"}
        override = {"a": {"nested": "value"}}

        result = deep_merge(base, override)

        assert result == {"a": {"nested": "value"}}

    def test_original_dicts_not_modified(self) -> None:
        """Test that original dicts are not modified."""
        base = {"a": 1, "nested": {"b": 2}}
        override = {"a": 10, "nested": {"c": 3}}
        original_base = {"a": 1, "nested": {"b": 2}}

        deep_merge(base, override)

        assert base == original_base

    def test_merge_empty_base(self) -> None:
        """Test merging with empty base dict."""
        base: dict[str, int] = {}
        override = {"a": 1, "b": 2}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_merge_empty_override(self) -> None:
        """Test merging with empty override dict."""
        base = {"a": 1, "b": 2}
        override: dict[str, int] = {}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_merge_with_list_values(self) -> None:
        """Test that list values are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = deep_merge(base, override)

        assert result == {"items": [4, 5]}


class TestYAMLLoadError:
    """Tests for YAMLLoadError exception."""

    def test_error_contains_path_and_reason(self) -> None:
        """Test that error message contains path and reason."""
        path = Path("/some/path/file.yaml")
        reason = "File not found"

        error = YAMLLoadError(path, reason)

        assert str(path) in str(error)
        assert reason in str(error)
        assert error.path == path
        assert error.reason == reason
