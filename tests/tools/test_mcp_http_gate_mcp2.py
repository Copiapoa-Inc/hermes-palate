"""HTTP transport gate must accept either streamable-http client symbol.

mcp 2.0.0 removed the deprecated ``streamablehttp_client`` alias while
keeping ``streamable_http_client``. Gating ``_MCP_HTTP_AVAILABLE`` on the
legacy symbol alone silently parked every HTTP MCP server on mcp >= 2
installs (observed in the Palate fleet, 2026-08-01).
"""
import importlib
import sys
import types

import pytest


def _reload_mcp_tool_with_fake_streamable_http(monkeypatch, *, legacy: bool, new: bool):
    try:
        real_module = importlib.import_module("mcp.client.streamable_http")
    except ImportError:
        pytest.skip("mcp streamable_http transport not installed")
    fake = types.ModuleType("mcp.client.streamable_http")
    if legacy:
        fake.streamablehttp_client = getattr(
            real_module,
            "streamablehttp_client",
            getattr(real_module, "streamable_http_client", None),
        )
    if new:
        fake.streamable_http_client = getattr(
            real_module,
            "streamable_http_client",
            getattr(real_module, "streamablehttp_client", None),
        )
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", fake)
    import tools.mcp_tool as mcp_tool
    return importlib.reload(mcp_tool)


@pytest.fixture(autouse=True)
def _restore_mcp_tool():
    yield
    import tools.mcp_tool as mcp_tool
    importlib.reload(mcp_tool)


def test_new_only_install_keeps_http_available(monkeypatch):
    module = _reload_mcp_tool_with_fake_streamable_http(
        monkeypatch, legacy=False, new=True,
    )
    assert module._MCP_HTTP_AVAILABLE is True
    assert module._MCP_NEW_HTTP is True


def test_legacy_only_install_keeps_http_available(monkeypatch):
    module = _reload_mcp_tool_with_fake_streamable_http(
        monkeypatch, legacy=True, new=False,
    )
    assert module._MCP_HTTP_AVAILABLE is True
    assert module._MCP_NEW_HTTP is False


def test_no_streamable_http_symbols_disables_http(monkeypatch):
    module = _reload_mcp_tool_with_fake_streamable_http(
        monkeypatch, legacy=False, new=False,
    )
    assert module._MCP_HTTP_AVAILABLE is False
