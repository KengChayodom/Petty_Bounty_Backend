"""PromptPay identifiers. Feature 6 takes only a Thai mobile number, the same
rule the Flutter app applies before it builds a QR code."""
import re

_THAI_MOBILE = re.compile(r"^0[689]\d{8}$")


def is_thai_mobile(number: str | None) -> bool:
    """True when `number`, with spaces and dashes removed, is ten digits
    starting 06, 08 or 09."""
    if not number:
        return False
    return bool(_THAI_MOBILE.match(re.sub(r"[\s-]", "", number)))
