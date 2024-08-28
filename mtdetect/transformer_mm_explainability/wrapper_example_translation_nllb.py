
import sys
import subprocess

# Parallelize with parallel: cat sentences.txt | parallel --pipe -N 1 -j $n python3 wrapper.py

source_lang = sys.argv[1] if len(sys.argv) > 1 else "eng_Latn"
target_lang = sys.argv[2] if len(sys.argv) > 2 else "deu_Latn"
beam_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
beam_size = str(beam_size)
device = sys.argv[4] if len(sys.argv) > 4 else ''

for idx, l in enumerate(sys.stdin):
    l = l.rstrip("\r\n").split('\t')
    l1 = l[0]
    l2 = l[1] if len(l) > 1 else ''

    execution_result = subprocess.run(["python3", "mtdetect/transformer_mm_explainability/example_translation_nllb.py",
                                       l1, l2, source_lang, target_lang, beam_size, device],
                                       capture_output=True, text=True)
    stdout = execution_result.stdout
    warning_count = stdout.count("warning: ")

    print(f"wrapper_result\t{idx}\t{execution_result.returncode}\t{warning_count}\t{l1}\t{l2}")

    if idx % 10 == 0:
        sys.stdout.flush()
