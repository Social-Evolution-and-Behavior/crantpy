# -*- coding: utf-8 -*-
"""
This module contains custom exceptions for crantpy.
"""
from typing import Optional


class NoMatchesError(ValueError):
    """Raised if no matches are found.

    Parameters
    ----------
    message : str, optional
        The error message.
    """

    def __init__(self, message: Optional[str] = None) -> None:
        self.message = message if message else "No matches found"
        super().__init__(self.message)


class FilteringError(ValueError):
    """Raised if a filtering operation fails.

    Parameters
    ----------
    message : str, optional
        The error message.
    """

    def __init__(self, message: Optional[str] = None) -> None:
        self.message = message if message else "Filtering operation failed"
        super().__init__(self.message)
