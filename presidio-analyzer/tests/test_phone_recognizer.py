import pytest

from presidio_analyzer.predefined_recognizers.generic.phone_recognizer import PhoneRecognizer
from tests import assert_result, assert_result_with_textual_explanation


@pytest.fixture(scope="module")
def recognizer():
    return PhoneRecognizer(
        supported_regions=PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS + ("JP", "CN")
    )



@pytest.mark.parametrize(
    "text, expected_len, entities, expected_positions, score",
    [
        # fmt: off
        ("My US number is (415) 555-0132, and my international one is +1 415 555 0132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 75),), 0.4),
        ("My Israeli number is 09-7625400", 1, ["PHONE_NUMBER"], ((21, 31), ), 0.4),
        ("_: (415)555-0132", 1, ["PHONE_NUMBER"], ((3, 16), ), 0.4),
        ("United States: (415)555-0132", 1, ["PHONE_NUMBER"], ((15, 28), ), 0.4),
        ("US: 415-555-0132", 1, ["PHONE_NUMBER"], ((4, 16), ), 0.4),  # 'us' stop word
        ("_: +55 11 98456 5666", 1, ["PHONE_NUMBER"], ((3, 20), ), 0.4),
        ("Brazil: +55 11 98456 5666", 1, ["PHONE_NUMBER"], ((8, 25), ), 0.4),
        ("BR: +55 11 98456 5666", 1, ["PHONE_NUMBER"], ((4, 21), ), 0.4),
        ("My Japanese number is 090-1234-5678", 1, ["PHONE_NUMBER"],((22, 35), ), 0.4),
        ("My CN number is 13812345678", 1, ["PHONE_NUMBER"],((16, 27), ), 0.4),
        # GB national-format number: only matched when region is the valid ISO code
        # "GB" (not "UK"). Regression test for the DEFAULT_SUPPORTED_REGIONS fix.
        ("My UK number is 020 7946 0958", 1, ["PHONE_NUMBER"], ((16, 29), ), 0.4),
        # fmt: on
    ],
)
def test_when_all_phones_then_succeed(
    spacy_nlp_engine,
    text,
    expected_len,
    entities,
    expected_positions,
    score,
    recognizer,
):
    nlp_artifacts = spacy_nlp_engine.process_text(text, "en")
    results = recognizer.analyze(text, entities, nlp_artifacts=nlp_artifacts)
    assert len(results) == expected_len
    for i, (res, (st_pos, fn_pos)) in enumerate(zip(results, expected_positions)):
        assert_result(res, entities[i], st_pos, fn_pos, score)


@pytest.mark.parametrize(
    "text, expected_len, entities, expected_positions, score, leniency",
    [
        # fmt: off
        ("My US number is (415) 555-0132, and my international one is415-555-0132",
         1, ["PHONE_NUMBER"], ((16, 30), ), 0.4, 1),
        ("My US number is (415) 555-0132, and my international one is415-555-0132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"], ((16, 30), (59, 71), ), 0.4, 0),

        ("My US number is (415) 555-0132, and my international one is 91-415-555-0132",
         1, ["PHONE_NUMBER"], ((16, 30), ), 0.4, 2),
        ("My US number is (415) 555-0132, and my international one is 91-415-555-0132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"], ((16, 30), (60, 75), ), 0.4, 1),

        ("My US number is (415) 555-0132, and my international one is +91 4155 550132",
         1, ["PHONE_NUMBER"], ((16, 30), ), 0.4, 3),
        ("My US number is (415) 555-0132, and my international one is +91 4155 550132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"], ((16, 30), (60, 75), ), 0.4, 2),

        ("My US number is (415) 555-0132, and my international one is +91 4155550132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"], ((16, 30), (60, 74), ), 0.4, 3),
        # fmt: on
    ],
)
def test_when_phone_with_leniency_then_succeed(
    spacy_nlp_engine,
    text,
    expected_len,
    entities,
    expected_positions,
    score,
    leniency,
):
    nlp_artifacts = spacy_nlp_engine.process_text(text, "en")
    recognizer = PhoneRecognizer(leniency=leniency)
    results = recognizer.analyze(text, entities, nlp_artifacts=nlp_artifacts)
    assert len(results) == expected_len
    for i, (res, (st_pos, fn_pos)) in enumerate(zip(results, expected_positions)):
        assert_result(res, entities[i], st_pos, fn_pos, score)


@pytest.mark.parametrize(
    "text, expected_len, entities, expected_positions, score, expected_textual_explanations",
    [
        # fmt: off
        ("My US number is (415) 555-0132, and my international one is +44 (20) 7123 4567",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 78),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as GB region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +91 4155550132",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 74),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as IN region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +55 11 98456 5666",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 77),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as BR region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +49 30 1234567",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 74),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as DE region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +39 06 678 4343",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 75),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as IT region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +30 21 0 1234567",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 76),), 0.4, 
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as GR region phone number, using PhoneRecognizer']),
         ("My US number is (415) 555-0132, and my international one is +33 1 42 68 53 00",
         2, ["PHONE_NUMBER", "PHONE_NUMBER"],
         ((16, 30), (60, 77),), 0.4,
         ['Recognized as US region phone number, using PhoneRecognizer','Recognized as FR region phone number, using PhoneRecognizer']),
        # fmt: on
    ],
)
def test_when_phone_with_textual_explanation_then_succeed(
    spacy_nlp_engine,
    text,
    expected_len,
    entities,
    expected_positions,
    score,
    expected_textual_explanations,
):
    nlp_artifacts = spacy_nlp_engine.process_text(text, "en")
    recognizer = PhoneRecognizer()
    results = recognizer.analyze(text, entities, nlp_artifacts=nlp_artifacts)
    assert len(results) == expected_len
    for i, (res, (st_pos, fn_pos)) in enumerate(zip(results, expected_positions)):
        assert_result_with_textual_explanation(res, entities[i], st_pos, fn_pos, score, expected_textual_explanations[i])

def test_get_analysis_explanation():
    phone_recognizer = PhoneRecognizer()
    test_region = "US"
    explanation = phone_recognizer._get_analysis_explanation(test_region)
    assert explanation.recognizer == "PhoneRecognizer"

def test_get_supported_entities():
    default_phone_recognizer = PhoneRecognizer()
    default_supported_entities = default_phone_recognizer.get_supported_entities()
    assert default_supported_entities == ["PHONE_NUMBER"]

    entity_name = "TELEPHONE_OR_FAX"
    configured_phone_recognizer = PhoneRecognizer(supported_entity=entity_name)
    configured_supported_entities = configured_phone_recognizer.get_supported_entities()
    assert configured_supported_entities == [entity_name]


@pytest.mark.parametrize(
    "regions, text, expected_positions, expected_textual_explanations",
    [
        # fmt: off
        # Control: a single national-format number is already explained correctly.
        (["US"], "call (415) 555-0132 please",
         ((5, 19), ),
         ["Recognized as US region phone number, using PhoneRecognizer"]),
        # A recognizer restricted to "US" never runs a GB matcher, so "GB" can only
        # come from the international match. It was then reported for the US number.
        (["US"], "+44 20 7946 0958 or (415) 555-0132 now",
         ((0, 16), (20, 34)),
         ["Recognized as GB region phone number, using PhoneRecognizer",
          "Recognized as US region phone number, using PhoneRecognizer"]),
        # The leak is not limited to the match right after the international one:
        # every later match in the same region iteration inherits it.
        (["US"], "(415) 555-0132 and +44 20 7946 0958 and (212) 555-0187",
         ((0, 14), (19, 35), (40, 54)),
         ["Recognized as US region phone number, using PhoneRecognizer",
          "Recognized as GB region phone number, using PhoneRecognizer",
          "Recognized as US region phone number, using PhoneRecognizer"]),
        # Mirror image: a GB-only recognizer explains its own GB number as US.
        (["GB"], "+1 212 555 0187 or 020 7946 0958",
         ((0, 15), (19, 32)),
         ["Recognized as US region phone number, using PhoneRecognizer",
          "Recognized as GB region phone number, using PhoneRecognizer"]),
        # Same two numbers as test_when_phone_with_textual_explanation_then_succeed,
        # in the opposite order: which number is explained may not depend on the
        # order they appear in.
        (list(PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS),
         "My international number is +44 (20) 7123 4567 and my US one is (415) 555-0132",
         ((27, 45), (63, 77)),
         ["Recognized as GB region phone number, using PhoneRecognizer",
          "Recognized as US region phone number, using PhoneRecognizer"]),
        # fmt: on
    ],
)
def test_when_phone_region_is_reported_then_only_for_its_own_match(
    spacy_nlp_engine,
    regions,
    text,
    expected_positions,
    expected_textual_explanations,
):
    """Each result's explanation must name the region of its own match.

    analyze() reassigned the ``region`` loop variable to the region of the last
    number it could parse without a default region, so a later national-format
    number - which cannot be parsed that way, and so falls into the
    NumberParseException branch - was explained with the previous match's
    region.
    """
    nlp_artifacts = spacy_nlp_engine.process_text(text, "en")
    recognizer = PhoneRecognizer(supported_regions=regions)
    results = recognizer.analyze(text, ["PHONE_NUMBER"], nlp_artifacts=nlp_artifacts)

    assert len(results) == len(expected_positions)
    for result, (start, end), explanation in zip(
        results, expected_positions, expected_textual_explanations
    ):
        assert_result_with_textual_explanation(
            result, "PHONE_NUMBER", start, end, 0.4, explanation
        )
