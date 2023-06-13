import os
from wikinews import read_file, write_file

if __name__ == "__main__":

    dirpath = "data/GPT-4"
    savepath = "data/WikiNewsPred_GPT4.combined.txt"
    domains = list({'culture': 'ثقافة', 'health': 'صحة', 'politics': 'سياسة', 'science': 'علوم', 'sports': 'رياضة', 'art': 'فن', 'economics': 'اقتصاد'}.keys())

    paths = [os.path.join(dirpath, f"WikiNews.{domain}_temp=0.7.2-20.pred.combined") for domain in domains] 

    write_file(savepath, ['\n'.join(read_file(path)) for path in paths])
