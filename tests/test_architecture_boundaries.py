"""
Layer boundaries, checked by reading the source rather than trusting it.

Each rule below was broken at least once before it was written down, so it is
pinned the way test_route_auth_guard.py pins the login guard: a new file that
breaks it fails here, in CI, instead of in a review that might miss it.

  1. Only app/repositories/ talks to supabase-py (`.table`, `.rpc`, `.storage`).
  2. Only the composition points name a concrete `Supabase*Repository`:
     app/api/dependencies.py for requests, app/core/auth.py for the admin gate.
  3. A route never calls a repository itself. It asks for a service.
"""
import ast
import pathlib
import re

APP = pathlib.Path(__file__).resolve().parent.parent / "app"

# Places allowed to construct an adapter. Anything else must receive a port.
COMPOSITION_POINTS = {
    APP / "api" / "dependencies.py",
    APP / "core" / "auth.py",
}

_VENDOR_CALL = re.compile(r"\.(table|rpc)\(|\.storage\.from_\(")
_ADAPTER_NAME = re.compile(r"\bSupabase\w+Repository\b")


def _source_files(*parts):
    return sorted((APP.joinpath(*parts)).rglob("*.py"))


def _code_lines(path):
    """Source lines with comments and docstrings blanked, so prose that names
    a call (as this project's docstrings often do) is not mistaken for one."""
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(
            getattr(node, "value", None), ast.Constant
        ) and isinstance(node.value.value, str):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = ""
    return [
        (n, line.split("#", 1)[0])
        for n, line in enumerate(lines, start=1)
    ]


def test_only_repositories_call_supabase_py():
    offenders = [
        f"{path.relative_to(APP.parent)}:{n}"
        for path in _source_files()
        if APP / "repositories" not in path.parents
        and path != APP / "core" / "database.py"
        for n, line in _code_lines(path)
        if _VENDOR_CALL.search(line)
    ]
    assert offenders == [], f"supabase-py used outside app/repositories/: {offenders}"


def test_only_composition_points_construct_adapters():
    offenders = [
        f"{path.relative_to(APP.parent)}:{n}"
        for path in _source_files()
        if APP / "repositories" not in path.parents
        and path not in COMPOSITION_POINTS
        and path != APP / "main.py"  # imports nothing concrete; kept explicit
        for n, line in _code_lines(path)
        if _ADAPTER_NAME.search(line)
    ]
    assert offenders == [], f"concrete adapter named outside a composition point: {offenders}"


def test_routes_never_call_a_repository_directly():
    """`repo.<method>(` in a route is business logic (or at least a blocking
    DB call) that belongs in a service."""
    offenders = [
        f"{path.relative_to(APP.parent)}:{n}"
        for path in _source_files("api")
        if path.name != "dependencies.py"
        for n, line in _code_lines(path)
        if re.search(r"\b\w*repo\.\w+\(", line)
    ]
    assert offenders == [], f"route calls a repository directly: {offenders}"
