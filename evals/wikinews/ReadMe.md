# Steps for Overlapping Context Window Evaluation


## Init Files
0. `python split_wikinews.py` to get a file for each domain.

## For Each Domain
1. `python segment.py --domain {domain}` on the text file. You can add file template you wish to segment along with the context window and stride in the `configs/segment.yaml` file.
2. `python wikinews_overlap.py {domain}`
3. `python clean_preds.py {domain}` to remove empty new lines which can happen when pausing then re-running the script.
4. `python check_discrepancy.py {domain}`
5. `python combine_segments.py {domain}`