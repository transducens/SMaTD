
import sys
import subprocess

# Parallelize with parallel: cat sentences.txt | parallel --pipe -N 1 -j $n python3 wrapper.py

source_lang = sys.argv[1] if len(sys.argv) > 1 else "eng_Latn"
target_lang = sys.argv[2] if len(sys.argv) > 2 else "deu_Latn"

for idx, l in enumerate(sys.stdin):
    l = l.rstrip("\r\n").split('\t')
    l1 = l[0]
    l2 = l[1] if len(l) > 1 else ''

    execution_result = subprocess.run(["python3", "mtdetect/transformer_mm_explainability/example_translation_nllb.py", l1, l2, source_lang, target_lang])

    print(f"wrapper_result\t{idx}\t{execution_result.returncode}\t{l1}\t{l2}")
