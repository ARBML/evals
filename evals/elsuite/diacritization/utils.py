from diacritization_evaluation import wer, der
import re

def post_process(txt):
  puncts = '؟،-[]:؛;()$#@&+=_-{}"\'.\n/\\0123456789١٢٣٤٥٦٧٨٩٠١'
  out = ""
  for c in txt:
    if c in puncts:
      out += ' '
      continue
    out += c
  return re.sub(' +', ' ', out).strip()


def calculate_diacritization_score(predicted, expected):
    # remove punctuations and special characters
    predicted = post_process(predicted)

    der_ = der.calculate_der(
        expected,
        predicted,
        case_ending=True
    )
    wer_ = wer.calculate_wer(
        expected,
        predicted,
        case_ending=True,
    )
    der_no_ce = der.calculate_der(
        expected,
        predicted,
        case_ending=False,
    )
    wer_no_ce = wer.calculate_wer(
        expected,
        predicted,
        case_ending=False,
    )
    return der_, wer_, der_no_ce, wer_no_ce
