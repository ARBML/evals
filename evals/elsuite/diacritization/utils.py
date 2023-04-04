from diacritization_evaluation import wer, der


def calculate_diacritization_score(predicted, expected):
    der_score = der.calculate_der(
        expected,
        predicted,
    )
    wer_score = wer.calculate_wer(
        expected,
        predicted,
        case_ending=False,
    )
    der_score_with_case_ending = der.calculate_der(
        expected,
        predicted,
        case_ending=True,
    )
    wer_score_with_case_ending = wer.calculate_wer(
        expected,
        predicted,
        case_ending=True,
    )
    return der_score, wer_score, der_score_with_case_ending, wer_score_with_case_ending
