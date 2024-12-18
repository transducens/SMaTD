
import re
import sys
import unicodedata

from sacremoses import MosesPunctNormalizer

# https://github.com/facebookresearch/stopes/blob/fff57bcf3ba9a32cfb49163da2736652f0ab56f3/stopes/pipelines/monolingual/utils/remove_regex.py#L61
def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

# https://github.com/facebookresearch/stopes/blob/fff57bcf3ba9a32cfb49163da2736652f0ab56f3/stopes/pipelines/monolingual/monolingual_line_processor.py#L202
class SentenceSplitClean:
#    def __init__(self, splitter_lang: str, split_algo: str):
    def __init__(self):
        # setup sentence splitter
#        self.splitter = get_split_algo(splitter_lang, split_algo=split_algo)

        # setup "moses" normalization
        self.mpn = MosesPunctNormalizer(lang="en")
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
#        self.replace_nonprint = remove_regex.get_non_printing_char_replacer(" ")
        self.replace_nonprint = get_non_printing_char_replacer(" ")

    def __call__(self, sentence_splits):
        assert isinstance(sentence_splits, list), type(sentence_splits)

#        sentence_splits = self.splitter(line)
#        line_hash = xxhash.xxh3_64_intdigest(line)

        for sent in sentence_splits:
            # normalize -- moses equivalent
            clean = self.mpn.normalize(sent)
            clean = self.replace_nonprint(clean)
            # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
            clean = unicodedata.normalize("NFKC", clean)

#            yield (line_hash, sent, clean)
            yield sent, clean

norm = SentenceSplitClean()

def get_clean_sentences_generator(sentences):
    for sent, clean in norm(sentences):
        yield clean

def get_clean_sentence(s):
    assert isinstance(s, str)

    for sent, clean in norm([s]):
        return clean

if __name__ == "__main__":
    print("sent == clean\tsent\tclean")

    for l in sys.stdin:
        l = l.rstrip("\r\n")

        for sent, clean in norm([l]):
            print(f"{sent == clean}\t{sent}\t{clean}")
