"""Norwegian fødselsnummer recognizer."""

from __future__ import annotations

import datetime
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class NoFodselsnummerRecognizer(PatternRecognizer):
    """Recognizes Norwegian fødselsnummer (national identity numbers).

    A fødselsnummer is eleven digits: DDMMYY, a three-digit individual number
    and two check digits. The check digits use two independent mod-11 sums,
    so a generic eleven-digit number is not enough.

    Reference:
    https://www.skatteetaten.no/en/person/national-registry/birth-and-name-selection/children-born-in-norway/national-id-number/
    """

    COUNTRY_CODE = "no"

    PATTERNS = [
        Pattern(
            "Norwegian Fødselsnummer (Weak)",
            r"\b\d{11}\b",
            0.3,
        ),
    ]

    CONTEXT = [
        "fødselsnummer",
        "fodselsnummer",
        "fødselsdato",
        "personnummer",
        "fnr",
        "national id",
    ]

    _CHECK_1_WEIGHTS = (3, 7, 6, 1, 8, 9, 4, 5, 2)
    _CHECK_2_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "no",
        supported_entity: str = "NO_FODSELSNUMMER",
        name: Optional[str] = None,
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    @staticmethod
    def _check_digit(digits: str, weights: tuple[int, ...]) -> Optional[int]:
        total = sum(int(digit) * weight for digit, weight in zip(digits, weights))
        remainder = total % 11
        if remainder == 0:
            return 0
        candidate = 11 - remainder
        return candidate if candidate < 10 else None

    def validate_result(self, pattern_text: str) -> bool:
        """Validate the date component and both mod-11 check digits."""
        digits = "".join(character for character in pattern_text if character.isdigit())
        if len(digits) != 11:
            return False

        day = int(digits[0:2])
        month = int(digits[2:4])
        year = int(digits[4:6])

        # The century is selected by the individual number, but any valid
        # day/month in either century is enough for PII detection.
        valid_date = False
        for full_year in (1900 + year, 2000 + year):
            try:
                datetime.date(full_year, month, day)
            except ValueError:
                continue
            valid_date = True
            break
        if not valid_date:
            return False

        first = self._check_digit(digits[0:9], self._CHECK_1_WEIGHTS)
        if first is None or first != int(digits[9]):
            return False

        second = self._check_digit(digits[0:10], self._CHECK_2_WEIGHTS)
        return second is not None and second == int(digits[10])
