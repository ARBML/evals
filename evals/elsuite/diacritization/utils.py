from diacritization_evaluation import wer, der


def calculate_diacritization_score(
    predicted,
    expected,
    case_ending=True,
):
    der_score = der.calculate_der(
        expected,
        predicted,
        case_ending=case_ending,
    )
    wer_score = wer.calculate_wer(
        expected,
        predicted,
        case_ending=case_ending,
    )
    return der_score, wer_score
