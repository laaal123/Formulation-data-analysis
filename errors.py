"""Typed refusals.

A Refusal means the app understood the request and is declining because the data
cannot support it. It is a normal, correct outcome and must never be caught and
converted into a number.
"""


class Refusal(Exception):
    """The analysis cannot be done on this data, with a stated reason."""


class DataError(Exception):
    """The input is malformed in a way the user must fix."""
