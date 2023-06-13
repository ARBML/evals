import os
import sys
import numpy as np
import pickle as pkl
from pyarabic.araby import strip_tashkeel, tokenize
from wikinews import read_file, read_json, write_file
from diac_eval import clear_line, CONSTANTS_PATH

DOMAIN = "art"

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        DOMAIN = sys.argv[1]
        print(f"> Domain: {DOMAIN}")
        
    dirpath = f"/Users/bkhmsi/Desktop/WikiNews/{DOMAIN}-overlap"
    dirpath = "./data/GPT-4"

    pred_path = os.path.join(dirpath, f"{DOMAIN}_temp=0.7.2-20.pred.clean")

    grnd_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.2-20.txt")
    grnd_path = os.path.join("data", f"{DOMAIN}-overlap", f"WikiNews.{DOMAIN}.2-20.txt")

    pmod_path = os.path.join(dirpath, f"{DOMAIN}_temp=0.7.2-20.pred.mod")

    preds = read_file(pred_path)
    grndt = read_file(grnd_path)

    new_preds = []

    for idx, (pred, grnd) in enumerate(zip(preds, grndt)):
        pred = pred.strip()
        grnd = grnd.strip()

        tpred = tokenize(pred)
        spred = tokenize(pred, morphs=strip_tashkeel)
        sgrnd = tokenize(grnd, morphs=strip_tashkeel)

        assert len(tpred) == len(spred)

        mod_tokens = []
        for j, tok in enumerate(sgrnd):
            if tok in spred:
                jdx = spred.index(tok)
                mod_tokens += [tpred[jdx]]
            else:
                print(tok)
                mod_tokens += [tok]
            
        new_preds += [' '.join(mod_tokens)]

    write_file(pmod_path, new_preds)