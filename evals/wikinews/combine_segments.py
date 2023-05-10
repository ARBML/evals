import os
import yaml
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import Counter, deque
from pyarabic.araby import tokenize, strip_tashkeel
from wikinews import read_file, write_file

diacritics = {
    "FATHA": 1,
    "KASRA": 2,
    "DAMMA": 3,
    "SUKUN": 4
}

DOMAIN = "sports"

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pkl.load(file)
    return data

def load_file_clean(base_path, domain, strip=False):
    f_name = os.path.join(base_path, f"WikiNews.ff{domain}.grnd")
    with open(f_name, 'r', encoding="utf-8") as fin:
        original_lines = [strip_tashkeel(preprocess(line)) if strip else preprocess(line) for line in fin.readlines()]
    return original_lines

def preprocess(line):
    return ' '.join(tokenize(line.strip()))

class DataAggregator:
    def __init__(self, config):
        self.base_path = config["paths"]["base"]
        self.special_tokens = ['<pad>', '<unk>', '<num>', '<punc>'] 
        self.delimeters = config["sentence-break"]["delimeters"]
        self.load_constants(config["paths"]["constants"])

        self.stride = config["sentence-break"]["stride"]  
        self.window = config["sentence-break"]["window"]  

    def load_constants(self, path):
        self.numbers = [c for c in "0123456789"]
        self.letter_list = self.special_tokens + list(load_pickle(os.path.join(path, 'ARABIC_LETTERS_LIST.pickle')))
        self.diacritic_list = [' '] + list(load_pickle(os.path.join(path, 'DIACRITICS_LIST.pickle')))

    def load_mapping(self, domain):
        mapping = {}
        file_ext = f"{self.stride}-{self.window}.filter.map"
        f_name = os.path.join(self.base_path, f"WikiNews.f{domain}.{file_ext}")
        with open(f_name, 'r') as fin:
            for line in fin:
                sent_idx, seg_idx, t_idx, c_idx = map(int, line.split(','))
                if sent_idx not in mapping:
                    mapping[sent_idx] = {}
                if seg_idx not in mapping[sent_idx]:
                    mapping[sent_idx][seg_idx] = {}
                if t_idx not in mapping[sent_idx][seg_idx]:
                    mapping[sent_idx][seg_idx][t_idx] = []
                mapping[sent_idx][seg_idx][t_idx] += [c_idx]
        return mapping

    def char_type(self, char):
        if char in self.letter_list:
            return self.letter_list.index(char)
        elif char in self.numbers:
            return self.letter_list.index('<num>')
        elif char in self.delimeters:
            return self.letter_list.index('<punc>')
        else:
            return self.letter_list.index('<unk>')   
        
    def create_labels(self, char):
        remap_dict = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4}
        char = [char[0]] + list(set(char[1:]))
        if len(char) > 3:
            char = char[:2] if self.diacritic_list[8] not in char else char[:3]

        char_idx = self.char_type(char[0])
        if len(char) == 1:
            return char_idx, 0, [remap_dict[0], 0, 0]
        elif len(char) == 2:  # If not shadda
            diacritic_index = self.diacritic_list.index(char[1])
            if diacritic_index in [2, 4, 6]:  # list of doubles
                return char_idx, diacritic_index, [remap_dict[diacritic_index - 1], 1, 0]
            elif diacritic_index == 8:
                return char_idx, diacritic_index, [0, 0, 1]
            else:
                return char_idx, diacritic_index, [remap_dict[diacritic_index], 0, 0]
        elif len(char) == 3:  # If shadda
            if self.diacritic_list[8] == char[1]:
                diacritic_index = self.diacritic_list.index(char[2])
            else:
                diacritic_index = self.diacritic_list.index(char[1])

            if diacritic_index in [2, 4, 6]:  # list of doubles
                return char_idx, diacritic_index+8, [remap_dict[diacritic_index - 1], 1, 1]
            else:
                return char_idx, diacritic_index+8, [remap_dict[diacritic_index], 0, 1]
        return None, None, None

    def create_label_for_word(self, split_word_):
        word_label_x = []
        diac_label_x = []
        diac_label_y = []
        for character_ in split_word_:
            char_x, diac_x, diac_y = self.create_labels(character_)
            if char_x == None:
                print(split_word_)
                raise ValueError(char_x)
            word_label_x.append(char_x)
            diac_label_x.append(diac_x)
            diac_label_y.append(diac_y)
        return word_label_x, diac_label_x, diac_label_y

    def split_word_on_characters_with_diacritics(self, word):
        word_queue = deque(word)
        split_word_on_characters = []
        temp_string = [word_queue.popleft()]

        while len(word_queue) > 0:
            poping_left = word_queue.popleft()
            if poping_left not in self.diacritic_list:
                split_word_on_characters.append(temp_string)
                temp_string = [poping_left]
            else:
                temp_string += [poping_left]
        split_word_on_characters.append(temp_string)
        return split_word_on_characters
            
    def separate(self, lines):
        preds = {'haraka': [], 'shadda': [], 'tanween': []}
        for seg_idx, line in tqdm(enumerate(lines)):
        
            tokens = tokenize(line.strip())
       
            token_h, token_t, token_s = [], [], []
            for word in tokens:
                split_word = self.split_word_on_characters_with_diacritics(word)
                _, _, cy_3head = self.create_label_for_word(split_word)
                token_h += [[y[0] for y in cy_3head]]
                token_t += [[y[1] for y in cy_3head]]
                token_s += [[y[2] for y in cy_3head]]

            preds['haraka'].append(token_h)
            preds['tanween'].append(token_t)
            preds['shadda'].append(token_s)
    
        return (
            preds['haraka'],
            preds["tanween"],
            preds["shadda"],
        )


        
def shakkel_char(diac: int, tanween: bool, shadda: bool) -> str:
    returned_text = ""
    if shadda and diac != diacritics["SUKUN"]:
        returned_text += "\u0651"

    if diac == diacritics["FATHA"]:
        returned_text += "\u064E" if not tanween else "\u064B"
    elif diac == diacritics["KASRA"]:
        returned_text += "\u0650" if not tanween else "\u064D"
    elif diac == diacritics["DAMMA"]:
        returned_text += "\u064F" if not tanween else "\u064C"
    elif diac == diacritics["SUKUN"]:
        returned_text += "\u0652"

    return returned_text

if __name__ == "__main__":

    config_path = "./configs/segment_wikinews.yaml"
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataagg = DataAggregator(config)

    mapping = dataagg.load_mapping(DOMAIN)
    original_lines = load_file_clean(config["paths"]["base"], DOMAIN, strip=True)

    lines = read_file(os.path.join(config["paths"]["base"], f"WikiNews.f{DOMAIN}.2-20.fpred.mod"))
    
    seg_lines = read_file(os.path.join(config["paths"]["base"], f"WikiNews.f{DOMAIN}.2-20.ffilter.txt"))
    
    y_gen_diac, y_gen_tanween, y_gen_shadda = dataagg.separate(lines)
    
    diacritized_lines = []
    for sent_idx, line in tqdm(enumerate(original_lines), total=len(original_lines)):
        diacritized_line = ""
        for char_idx, char in enumerate(line):
            diacritized_line += char
            char_vote_haraka, char_vote_shadda, char_vote_tanween = [], [], []
            if sent_idx not in mapping: continue
            for seg_idx in mapping[sent_idx]:
                for t_idx in mapping[sent_idx][seg_idx]:                        
                    if char_idx in mapping[sent_idx][seg_idx][t_idx]:
                        try:
                            c_idx = mapping[sent_idx][seg_idx][t_idx].index(char_idx)
                            if y_gen_diac[seg_idx][t_idx][c_idx] != 0:
                                char_vote_haraka  += [y_gen_diac[seg_idx][t_idx][c_idx]]
                                char_vote_shadda  += [y_gen_shadda[seg_idx][t_idx][c_idx]]
                                char_vote_tanween += [y_gen_tanween[seg_idx][t_idx][c_idx]]
                        except:
                            gline_tokens = tokenize(lines[seg_idx].strip(), morphs=strip_tashkeel)
                            sline_tokens = tokenize(seg_lines[seg_idx].strip(), morphs=strip_tashkeel)
                            print(f"Error {seg_idx}/{t_idx}/{char_idx} --> {gline_tokens[t_idx]}")
                            breakpoint()

            if len(char_vote_haraka) > 0:
                char_mv_diac = Counter(char_vote_haraka).most_common()[0][0]
                char_mv_shadda = Counter(char_vote_shadda).most_common()[0][0]
                char_mv_tanween = Counter(char_vote_tanween).most_common()[0][0]
                diacritized_line += shakkel_char(char_mv_diac, char_mv_tanween, char_mv_shadda)
        
        diacritized_lines += [diacritized_line.strip()]
    write_file(os.path.join(config["paths"]["base"], f"WikiNews.{DOMAIN}.2-20.pred.combined"), diacritized_lines)

