from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    mark_expr = config.option.markexpr
    if "all_versions" in mark_expr:
        return
    selected = [item for item in items if "all_versions" not in item.keywords]
    deselected = [item for item in items if "all_versions" in item.keywords]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
