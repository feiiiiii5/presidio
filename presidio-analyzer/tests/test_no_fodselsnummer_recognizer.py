import pytest

from presidio_analyzer.predefined_recognizers import NoFodselsnummerRecognizer
from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    return NoFodselsnummerRecognizer()


@pytest.fixture(scope="module")
def entities():
    return ["NO_FODSELSNUMMER"]


@pytest.mark.parametrize(
    "text, expected_len, expected_position, expected_score",
    [
        # Valid date and both mod-11 check digits.
        ("12058545633", 1, (0, 11), 1.0),
        ("31129978913", 1, (0, 11), 1.0),
        # Same number inside a sentence.
        ("Fødselsnummer: 01010150074.", 1, (15, 26), 1.0),
        # Wrong second check digit.
        ("12058545634", 0, (), ()),
        # Impossible date (32 December).
        ("32129978913", 0, (), ()),
        # Eleven digits that are not a date at all.
        ("12345678901", 0, (), ()),
    ],
)
def test_when_fodselsnummer_in_text_then_valid_ones_are_found(
    text,
    expected_len,
    expected_position,
    expected_score,
    recognizer,
    entities,
):
    results = recognizer.analyze(text, entities)

    assert len(results) == expected_len
    if results:
        assert_result(
            results[0],
            entities[0],
            expected_position[0],
            expected_position[1],
            expected_score,
        )
