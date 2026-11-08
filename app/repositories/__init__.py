"""Repository seams — interfaces owned by this codebase that wrap the
supabase-py client. Services depend on these Protocols, never on the client
directly. Every `.table(...)`, `.rpc(...)`, `.execute()` lives under this
package (grep-able invariant per TEST_PLAN §6 / db-testing-seams §7).
"""
