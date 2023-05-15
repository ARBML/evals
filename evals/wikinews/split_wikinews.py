import os
import numpy as np
from pyarabic.araby import strip_tashkeel
from torch.utils.data import Dataset

def read_file(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = fin.readlines()
    return data 

def write_file(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(data))


class WikiNews(Dataset):
    def __init__(self, dirpath, diac=False) -> None:
        filename = "WikiNewsTruth.txt.diac" if diac else "WikiNewsTruth.txt"
        self.data_domain = {}
        self.cluster_by_domain(read_file(os.path.join(dirpath, filename)))
        self.domain_map = {'culture': 'ثقافة', 'health': 'صحة', 'politics': 'سياسة', 'science': 'علوم', 'sports': 'رياضة', 'art': 'فن', 'economics': 'اقتصاد'}
        self.set_domain("sports")
        self.set_instruction("Please diacritize the following Arabic sentence")

    def set_domain(self, domain):
        self.domain = self.domain_map[domain]

    def set_instruction(self, instruction):
        self.instruction = instruction

    def cluster_by_domain(self, data):
        for line in data:
            line = line.strip()
            if line[0] == "#":
                last_domain = strip_tashkeel(line[1:].strip())
                self.data_domain[last_domain] = []
            else:
                self.data_domain[last_domain] += [line]

    def getdata(self, max_n=None):
        max_n = max_n or len(self.data_domain[self.domain])
        return self.data_domain[self.domain][:max_n]
    
    def stats(self):
        print(f"# Lines: {len(self)}")
        num_words = []
        for line in self.data_domain[self.domain]:
            num_words += [len(line.split())]
        print(f"Mean: {np.mean(num_words)} | Stdev: {np.std(num_words)}")
        print(f"Max: {np.max(num_words)} | Min: {np.min(num_words)}")


    def __getitem__(self, index):
        line = self.data_domain[self.domain][index]
        payload = {"role": "user", "content": f"{self.instruction}:\n{line}"}
        return payload
    
    def __len__(self):
        return len(self.data_domain[self.domain])


if __name__ == "__main__":

    dirpath = "/Users/bkhmsi/Desktop/WikiNews"
    dataset = WikiNews(dirpath, diac=True)

    for domain in dataset.domain_map:
        grnd_path = os.path.join(dirpath, f"WikiNews.{domain}.grnd")
        dataset.set_domain(domain)
        if not os.path.exists(grnd_path):
            write_file(grnd_path, dataset.getdata())
