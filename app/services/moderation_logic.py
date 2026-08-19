"""Pure moderation logic — no I/O, no DB, no framework.

The specs and the database disagree on spelling, and this module is where that
is reconciled once instead of at every call site:

  * UD-14 writes the flag reasons as "Spam", "Not_a_pet", "Inappropriate_image"
    while MD-39 writes them as "Spam", "Not a pet", "Inappropriate image". The
    `report_reason` enum only accepts the underscored form.
  * MD-40 writes the decisions as "Dismissed" and "Reviewed and banned", UD-14
    as "Dismiss Flag" / "Uphold and Ban User"; the `report_status` enum only
    accepts "Dismissed" and "Reviewed_Ban".

Both normalisers reject anything outside their set rather than guessing, so an
unknown value is a 400 at the edge and never reaches the database as a failed
enum cast.
"""

# report_reason enum values (sql-update.txt:14).
FLAG_REASONS = ("Spam", "Not_a_pet", "Inappropriate_image")

# report_status enum values (sql-update.txt:15). 'Pending' is the insert
# default; the two below are the terminal states an admin can write.
DECISION_DISMISS = "Dismissed"
DECISION_UPHOLD = "Reviewed_Ban"
FLAG_DECISIONS = (DECISION_DISMISS, DECISION_UPHOLD)

# The same enum read as a *queue filter*. Unlike FLAG_DECISIONS this includes
# 'Pending', because Pending is precisely the bucket an administrator reads the
# queue to find — it is not a decision, but it is a legitimate thing to ask for.
FLAG_STATUSES = ("Pending", DECISION_DISMISS, DECISION_UPHOLD)

_REASON_ALIASES = {
    "spam": "Spam",
    "not a pet": "Not_a_pet",
    "not_a_pet": "Not_a_pet",
    "notapet": "Not_a_pet",
    "inappropriate image": "Inappropriate_image",
    "inappropriate_image": "Inappropriate_image",
}

_DECISION_ALIASES = {
    "dismissed": DECISION_DISMISS,
    "dismiss": DECISION_DISMISS,
    "dismiss flag": DECISION_DISMISS,
    "reviewed and banned": DECISION_UPHOLD,
    "reviewed_ban": DECISION_UPHOLD,
    "uphold": DECISION_UPHOLD,
    "uphold and ban user": DECISION_UPHOLD,
}


def normalize_flag_reason(reason: str | None) -> str:
    """Map a caller-supplied reason onto the `report_reason` enum.

    Raises ValueError for anything outside the permitted set (MD-39's 400).
    """
    key = (reason or "").strip().lower()
    normalized = _REASON_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"reason must be one of {', '.join(FLAG_REASONS)}; got {reason!r}"
        )
    return normalized


def normalize_flag_decision(decision: str | None) -> str:
    """Map an admin decision onto the terminal `report_status` enum value.

    Raises ValueError for anything outside {Dismissed, Reviewed_Ban} — notably
    for "Pending", which is a starting state and not a decision.
    """
    key = (decision or "").strip().lower()
    normalized = _DECISION_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"decision must be one of {', '.join(FLAG_DECISIONS)}; "
            f"got {decision!r}"
        )
    return normalized


def normalize_flag_status_filter(status: str | None) -> str | None:
    """Map a moderation-queue filter onto the `report_status` enum.

    `None` passes through and means "every status", not "a status which is
    null" — the same convention MD-37's missing-pet browse uses. Any other
    unrecognised value raises ValueError (MD-51's 400) rather than reaching
    PostgREST and failing there as an enum cast, which would surface as a 500.
    Matching is case-insensitive but exact on the enum names; the decision
    aliases are deliberately not accepted, because "uphold" is an instruction
    and this argument is a filter.
    """
    if status is None:
        return None
    key = status.strip().lower()
    for value in FLAG_STATUSES:
        if value.lower() == key:
            return value
    raise ValueError(
        f"status must be one of {', '.join(FLAG_STATUSES)}; got {status!r}"
    )


def build_flag_payload(
    sighting_id: str, reason: str, reporter_id: str
) -> dict:
    """The `reports` INSERT contract for MD-39.

    `reporter_id` is always the verified JWT identity handed in by the route —
    the request body never carries it — and the status is always Pending, so a
    flag cannot be created pre-moderated.
    """
    return {
        "sighting_id": sighting_id,
        "reason": normalize_flag_reason(reason),
        "reporter_id": reporter_id,
        "status": "Pending",
    }
