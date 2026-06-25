from flowyforge.core.registry import PluginRegistry


def test_registry_register_get_list_clear() -> None:
    registry = PluginRegistry()
    plugin = object()

    registry.register("demo", plugin)

    assert registry.get("demo") is plugin
    assert registry.list() == ["demo"]

    registry.clear()

    assert registry.list() == []

