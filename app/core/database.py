"""
Database connection management for Supabase.

Provides a per-thread Supabase client (behind a process-wide proxy) and
dependency injection for FastAPI endpoints.

WHY PER-THREAD AND NOT A SINGLETON
----------------------------------
`supabase.Client` wraps a BLOCKING `httpx.Client`. Since the services started
offloading their blocking DB work with `asyncio.to_thread`, several worker
threads drive that client's connection pool at the same time. httpcore's sync
stream does `sock.settimeout(...)` and then `recv()` on a pooled socket, so when
two threads land on the same connection one of them gets `EAGAIN` instead of
blocking:

    httpx.ReadError: [Errno 35] Resource temporarily unavailable

which surfaced to users as an intermittent `500` (e.g. "Failed to retrieve
hunter stats"). Measured at 40-way concurrency x 12 rounds: 62/480 requests
failed with one shared client, 0/480 with a client per thread.

Each thread does strictly serial I/O, so a per-thread client keeps a pool of
about one connection — this costs roughly what the shared pool cost, without
the cross-thread race. Thread count is bounded by the two pools that call us
(anyio's FastAPI threadpool and the default `asyncio.to_thread` executor).

WHY A PROXY AND NOT JUST A THREAD-LOCAL GETTER
----------------------------------------------
FastAPI resolves `Depends(get_supabase_client)` on ONE thread and the service
then uses that object inside `asyncio.to_thread`, i.e. on a DIFFERENT thread.
A thread-local returned at dependency-resolution time would therefore still be
shared across threads. `_ThreadLocalClientProxy` defers the lookup to the
moment of use, so `db.table(...)` always runs on the calling thread's own
client. Call sites are unchanged.
"""
import threading
import weakref
from logging import getLogger
from typing import Annotated, Any

from fastapi import Depends
from supabase import create_client, Client

from app.core.config import settings

logger = getLogger(__name__)

# One real client per thread. The registry is WEAK on purpose: `threading.local`
# already drops a client when its thread dies, and a strong registry would pin
# every client of every retired thread in memory for the life of the process.
_thread_state = threading.local()
_all_clients: "weakref.WeakSet[Client]" = weakref.WeakSet()
_all_clients_lock = threading.Lock()


def _client_for_current_thread() -> Client:
    """Get or create the calling thread's own Supabase client."""
    client = getattr(_thread_state, "client", None)
    if client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
            )

        logger.info(
            "Creating Supabase client for thread %s", threading.current_thread().name
        )
        client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        _thread_state.client = client
        with _all_clients_lock:
            _all_clients.add(client)

    return client


class _ThreadLocalClientProxy:
    """
    Stands in for a `supabase.Client`, forwarding every attribute access to the
    client owned by whichever thread is making the call.

    This is what makes the per-thread design survive `asyncio.to_thread`: the
    proxy can be resolved as a FastAPI dependency on one thread and used on
    another, and the underlying socket work still happens on a client that no
    other thread touches.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_client_for_current_thread(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<ThreadLocalSupabaseClient threads={len(_all_clients)}>"


_supabase_proxy = _ThreadLocalClientProxy()


def get_supabase_client() -> Client:
    """
    Get the Supabase client accessor.

    Returns a proxy that routes each call to the calling thread's own client
    (using the service role key). Safe to hold across threads.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_KEY are not set
    """
    return _supabase_proxy  # type: ignore[return-value]


# Type alias for dependency injection
SupabaseDep = Annotated[Client, Depends(get_supabase_client)]


async def close_supabase_client() -> None:
    """
    Drop every per-thread Supabase client.

    This should be called on application shutdown.
    """
    with _all_clients_lock:
        live = len(_all_clients)
        if live:
            logger.info("Closing %d live Supabase client(s)", live)
            _all_clients.clear()
    _thread_state.client = None
