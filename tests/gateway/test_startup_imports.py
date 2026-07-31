"""Import-cost gates for the gateway startup path."""

from __future__ import annotations

import builtins
import sys
import types

import pytest


@pytest.mark.asyncio
async def test_mcp_discovery_does_not_import_mcp_stack_without_servers(
    monkeypatch,
    tmp_path,
):
    """MCP-free users must not pay to import the optional MCP runtime."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway import run as gateway_run

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tools.mcp_tool":
            raise AssertionError("MCP stack imported without configured servers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    await gateway_run._discover_mcp_tools_if_configured()


@pytest.mark.asyncio
async def test_mcp_discovery_still_runs_with_configured_servers(
    monkeypatch,
    tmp_path,
):
    """Configured MCP servers must keep the existing discovery behavior."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "mcp_servers:\n  local:\n    command: local-mcp\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway import run as gateway_run

    calls = []

    def discover_mcp_tools():
        calls.append("discover")

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=discover_mcp_tools),
    )

    await gateway_run._discover_mcp_tools_if_configured()

    assert calls == ["discover"]
