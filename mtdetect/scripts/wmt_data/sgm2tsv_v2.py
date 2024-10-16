
import re
import sys

p_setid = re.compile(r'^.* setid="([^"]*)".*>$')
p_trglang = re.compile(r'^.* trglang="([^"]*)".*>$')
p_docid = re.compile(r'^.* docid="([^"]*)".*>$')
p_genre = re.compile(r'^.* genre="([^"]*)".*>$')
p_origlang = re.compile(r'^.* origlang="([^"]*)".*>$')
p_seg = re.compile(r'^<seg id="([0-9]+)">(.*)</seg>$')

def process_file(fn, src_or_ref_expected):
    fd = open(fn, "rt")
    result = []
    l = next(fd).strip()

    #assert l.startswith("<refset"), l
    assert l.startswith("<refset ") or l.startswith("<srcset "), l

    src_or_ref = "ref" if l.startswith("<refset ") else "src"
    setid = p_setid.search(l)
    trglang = p_trglang.search(l)

    assert src_or_ref == src_or_ref_expected, f"{src_or_ref} but {src_or_ref_expected} was expected"

    assert len(setid.groups()) == 1, f"{setid.group(0)} | {l}"

    if trglang is None:
        trglang = '-'
    else:
        assert len(trglang.groups()) == 1, f"{trglang.group(0)} | {l}"
        trglang = trglang.group(1).replace('\t', ' ')

    setid = setid.group(1).replace('\t', ' ')
    break_ok = False
    document_open = False
    paragraph_open = False
    current_document = 0
    current_paragraph = 0
    current_sentence = 0
    doc_docid = "UNK"
    doc_genre = "UNK"
    doc_origlang = "UNK"
    disable_paragraphs = False # Some files does not have paragraphs... let's detect that case
    line = 2

    while True:
        l = next(fd).strip()

        if l in ("</refset>", "</srcset>"):
            break_ok = True
            break

        # Process
        if l == '':
            line += 1
            continue

        if l.lower().startswith("<doc"):
            assert l.count('<') == 1 and l.count('>') == 1, l

            current_document += 1

            document_open = True
            p_doc_docid = p_docid.search(l)
            p_doc_genre = p_genre.search(l)
            p_doc_origlang = p_origlang.search(l)

            if p_doc_docid and len(p_doc_docid.groups()) > 0:
                doc_docid = p_doc_docid.group(1).replace('\t', ' ')
            if p_doc_genre and len(p_doc_genre.groups()) > 0:
                doc_genre = p_doc_genre.group(1).replace('\t', ' ')
            if p_doc_origlang and len(p_doc_origlang.groups()) > 0:
                doc_origlang = p_doc_origlang.group(1).replace('\t', ' ')

            line += 1
            continue
        elif l.lower().startswith("</doc>"):
            assert l.lower() == "</doc>", l

            document_open = False
            doc_docid = "UNK"
            doc_genre = "UNK"
            doc_origlang = "UNK"
            current_sentence = 0
            current_paragraph = 0

            line += 1
            continue
        elif l.lower() in ("<p>", "<hl>", "<h1>"):
            paragraph_open = True
            current_paragraph += 1

            line += 1
            continue
        elif l.lower() in ("</p>", "</hl>", "</h1>", "</hl"):
            paragraph_open = False

            line += 1
            continue

        current_sentence += 1

        ## Print

        if document_open and not paragraph_open and not disable_paragraphs:
            disable_paragraphs = True

        if disable_paragraphs and paragraph_open:
            # You lied!
            raise Exception("Paragraphs are disabled but we found paragraphs!")

        if disable_paragraphs:
            assert current_paragraph == 0, current_paragraph
        else:
            assert paragraph_open

        assert document_open
        assert current_paragraph <= current_sentence, f"{current_paragraph} > {current_sentence}"
        assert l.lower().startswith("<seg id=\"") and l.lower().endswith("</seg>"), f"{fn}: {line}: {l}"

        l = list(l)
        l[1] = 's' # lower useless part to match regex
        l[2] = 'e'
        l[3] = 'g'
        l[5] = 'i'
        l[6] = 'd'
        l[-4] = 's'
        l[-3] = 'e'
        l[-2] = 'g'
        l = ''.join(l)

        seg = p_seg.search(l)

        assert len(seg.groups()) == 2, f"{seg.group(0)} | {l}"

        seg_id = int(seg.group(1).replace('\t', ' ').strip())
        seg = seg.group(2).replace('\t', ' ').strip()

        assert seg_id == current_sentence, f"{fn}: {line}: {seg_id} vs {current_sentence}" # Some segments does not have correctly assigned the id...

        if doc_origlang == "UNK" and fn.split('/')[-1] == "news-test2008-src.de.sgm":
            doc_origlang = "en" # manual fix of the data...

        result.append((setid, doc_origlang, trglang, doc_genre, doc_docid, current_document, current_paragraph, current_sentence, seg, seg_id))

        line += 1

    assert break_ok
    assert not paragraph_open
    assert not document_open

    fd.close()

    return result

def check(a, b, idx, equal=True):
    if equal:
        assert a[idx] == b[idx], f"{idx}: should be equal: {a[idx]} vs {b[idx]}"
    else:
        assert a[idx] != b[idx], f"{idx}: should be different: {a[idx]} vs {b[idx]}"

langs = sys.argv[1].split(':')
target_origlang = langs[0]
target_trglang = '-' if len(langs) == 1 else langs[1]

assert len(target_origlang) == 2 or target_origlang == '-', target_origlang
assert len(target_trglang) == 2 or target_trglang == '-', target_trglang

src_fn, ref_fn = sys.argv[2].split(':')
src_result = process_file(src_fn, "src")
ref_result = process_file(ref_fn, "ref")

assert len(src_result) == len(ref_result)

header = "src_filename\tref_filename\tsetid\tsrc_lang\tref_lang\tgenre\tdocid\tnumber_doc\tnumber_paragraph\tnumber_sentence\tsrc_sentence\tref_sentence"
nheader = len(header.split('\t'))

print(header)

for src, ref in zip(src_result, ref_result):
    for i in (0, 1, 3, 4, 5, 6, 7, 9):
        check(src, ref, i, equal=True)
    for i in (2,):
        check(src, ref, i, equal=False)

    assert len(src) + 2 == len(ref) + 2 == nheader # + 2: segid not used (i.e., -1); both fns (+2); both sentences are printed, but each variable contains just one (i.e., +1). -1 + 2 + 1 = + 2

    setid = src[0]
    doc_origlang = src[1]
    trglang = ref[2] # necessary to be ref!
    doc_genre = src[3]
    doc_docid = src[4]
    current_document = src[5]
    current_paragraph = src[6]
    current_sentence = src[7]
    src_seg = src[8]
    ref_seg = ref[8]

    if target_origlang != '-' and doc_origlang != target_origlang:
        continue
    if target_trglang != '-' and trglang != target_trglang:
        continue

    s = f"{src_fn}\t{ref_fn}\t{setid}\t{doc_origlang}\t{trglang}\t{doc_genre}\t{doc_docid}\t{current_document}\t{current_paragraph}\t{current_sentence}\t{src_seg}\t{ref_seg}"

    assert len(s.split('\t')) == nheader

    print(s)
