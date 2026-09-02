"""Pure role-assignment logic — no I/O, no DB, no framework.

The two edge normalisers MD-58 and MD-59 apply before any database call, so a
malformed address or an unrecognised role is a 400 at the edge and never reaches
the database as a failed enum cast or a pointless lookup.

`user_role` is only {user, admin}. It is the whole vocabulary of authorisation
in this product: there is no third role, no account-state column beside it, and
no way to suspend an account. Withdrawing the administrator role returns an
account to `user`, which is what every account is on sign-up.
"""

# user_role enum values (data_model.md, verified live).
ROLE_USER = "user"
ROLE_ADMIN = "admin"
USER_ROLES = (ROLE_USER, ROLE_ADMIN)

_ROLE_ALIASES = {
    "user": ROLE_USER,
    "admin": ROLE_ADMIN,
    "administrator": ROLE_ADMIN,
}


def normalize_role(role: str | None) -> str:
    """Return the `user_role` value for `role`, or raise ValueError (-> 400).

    Case-insensitive, and "administrator" is accepted for the spec's own wording
    of the actor. Anything else is refused rather than guessed: silently mapping
    an unknown role to `user` would read as a successful demotion.
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError(
            f"role is required and must be one of {', '.join(USER_ROLES)}"
        )
    normalized = _ROLE_ALIASES.get(role.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of {', '.join(USER_ROLES)}"
        )
    return normalized


def normalize_email(email: str | None) -> str:
    """Return the address to look one account up by, or raise ValueError.

    Trimmed and lower-cased, because GoTrue stores addresses lower-cased but a
    person typing one into the console will not.

    The shape check is deliberately minimal — one `@`, something either side of
    it, a dot in the domain. This is not address validation, which only delivery
    can actually do. It exists so that an obvious typo costs no database round
    trip, and so that MD-58 cannot be handed a wildcard or an empty string and
    turned into the account enumeration it is specified not to be.
    """
    if not isinstance(email, str) or not email.strip():
        raise ValueError("An email address is required.")
    candidate = email.strip().lower()
    local, sep, domain = candidate.partition("@")
    if not sep or not local or not domain or "." not in domain:
        raise ValueError(f"'{email}' is not a valid email address.")
    if any(ch.isspace() for ch in candidate) or candidate.count("@") > 1:
        raise ValueError(f"'{email}' is not a valid email address.")
    return candidate
