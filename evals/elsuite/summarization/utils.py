from rouge_score import rouge_scorer

def calculate_rouge_score(
    predicted,
    expected,
):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    scores = scorer.score(expected, predicted)
    return scores