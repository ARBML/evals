from diacritization_evaluation import wer, der
import re
from pyarabic.araby import strip_diacritics
import difflib

def match_diacritics(p_, e_):

    p = strip_diacritics(p_).split(" ")
    e = strip_diacritics(e_).split(" ")

    p_ = p_.split(" ")
    e_ = e_.split(" ")
    

    # initiate the Differ object
    d = difflib.Differ()

    diff = d.compare(e, p)
    matching = list(a for a in diff)
    out = []
    i = 0
    for a in matching:
        # print(a)
        if a.startswith("-"):
            out.append(e[i])
        elif a.startswith("+"):
            i += 1
        elif a.startswith(" "):
            out.append(p_[i])
            i += 1
    assert len(out) == len(e_) 
    return " ".join(e_), " ".join(out).strip()

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

    # pinalize words with typos by replacing them with original words
    predicted, expected = match_diacritics(predicted, expected)
    # print("finished matching")

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
