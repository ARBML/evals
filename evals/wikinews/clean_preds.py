import os
import sys
from pyarabic.araby import tokenize, strip_tashkeel
from wikinews import read_file, read_json, write_file

DOMAIN = "art"

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        DOMAIN = sys.argv[1]
        print(f"> Domain: {DOMAIN}")
        
    dirpath = f"/Users/bkhmsi/Desktop/WikiNews/{DOMAIN}-overlap"
    grnd_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.2-20.txt")
    pred_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp=0.7.2-20.pred")
    pred_clean_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}_temp=0.7.2-20.pred.clean")

    preds = read_file(pred_path)
    grndt = read_file(grnd_path)

    print(f"> Domain: {DOMAIN}")
    print(f"Preds: #{len(preds)} | Grndt: #{len(grndt)}")

    index = 1
    clean_preds = []
    for pred in preds:
        if pred.strip() != '':
            clean_preds += [pred.strip()]
            pred_tokens = tokenize(pred.strip(), morphs=strip_tashkeel)
            grnd_tokens = tokenize(grndt[index-1].strip(), morphs=strip_tashkeel)
            diff = abs(len(grnd_tokens)-len(pred_tokens))
            is_error = pred.strip() == "Error"
            if diff > 2 and not is_error:
                print(f"{index}\t{diff}")
            index += 1
    
    assert len(clean_preds) == len(grndt)
    write_file(pred_clean_path, clean_preds)