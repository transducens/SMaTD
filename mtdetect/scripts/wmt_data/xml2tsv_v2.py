
import re
import sys

p_setid = re.compile(r'^<dataset id="([^"]*)">$')
p_collection = re.compile(r'^<collection id="([^"]*)">$')
p_trglang = re.compile(r'^.* trglang="([^"]*)".*>$')
p_docid = re.compile(r'^<doc .*id="([^"]*)".*>$')
p_doctestsuite = re.compile(r'^<doc .*testsuite="([^"]*)".*>$')
p_origlang = re.compile(r'^<doc .*origlang="([^"]*)".*>$')
p_trglang = re.compile(r'^<... .*lang="([^"]*)".*>$')
p_seg = re.compile(r'^<seg id="([0-9]+)">(.*)</seg>$')
p_translator = re.compile(r'^<... .*translator="([^"]*)"')

def process_collection(data, collection_name, setid, src_or_ref):
    document_open = False
    paragraph_open = False
    src_or_ref_open = False
    current_document = 0
    current_paragraph = 0
    current_sentence = 0
    doc_docid = "UNK"
    doc_origlang = "UNK"
    doc_testsuite = "UNK"
    trglang = "UNK"
    translator = "UNK"
    disable_paragraphs = False # Some files does not have paragraphs... let's detect that case
    result = []

    # header printing was here before

    for l in data:
        l = l.strip()

        # Process
        if l == '':
            continue

        if l.lower().startswith(f"<src ") or l.lower().startswith(f"<ref "):
            assert trglang == "UNK", trglang
            assert not src_or_ref_open

            src_or_ref_open = True
            is_src_or_ref = "src" if l.lower().startswith(f"<src ") else "ref"
            trglang = p_trglang.search(l)
            translator = p_translator.search(l)
            #current_document += 1

            assert len(trglang.groups()) == 1, f"{trglang.group(0)} | {l}"

            trglang = trglang.group(1)

            if translator and len(translator.groups()) > 0:
                assert len(translator.groups()) == 1, translator.groups()
            else:
                translator = "UNK"

            if l.lower().startswith("<src "):
                # TODO ???
                assert trglang == doc_origlang, f"{trglang} vs {doc_origlang}"

                #if trglang != doc_origlang:
                #    sys.stderr.write(f"{doc_docid}: {trglang} vs {doc_origlang}\n")

            continue
        elif l.lower() in ("</src>", "</ref>"):
            assert trglang != "UNK", trglang
            assert src_or_ref_open

            src_or_ref_open = False
            trglang = "UNK"
            current_sentence = 0
            current_paragraph = 0
            translator = "UNK"

            continue
        elif l.lower().startswith("<doc "):
            assert l.count('<') == 1 and l.count('>') == 1, l

            current_document += 1

            document_open = True
            p_doc_docid = p_docid.search(l)
            p_doc_origlang = p_origlang.search(l)
            p_doc_testsuite = p_doctestsuite.search(l)

            if p_doc_docid and len(p_doc_docid.groups()) > 0:
                doc_docid = p_doc_docid.group(1).replace('\t', ' ')
            if p_doc_origlang and len(p_doc_origlang.groups()) > 0:
                doc_origlang = p_doc_origlang.group(1).replace('\t', ' ')
            if p_doc_testsuite and len(p_doc_testsuite.groups()) > 0:
                doc_testsuite = p_doc_testsuite.group(1).replace('\t', ' ')

            continue
        elif l.lower().startswith("</doc>"):
            assert l.lower() == "</doc>", l

            document_open = False
            doc_docid = "UNK"
            doc_origlang = "UNK"
            doc_testsuite = "UNK"
            current_sentence = 0
            current_paragraph = 0

            continue
        elif l.lower() in ("<p>", "<hl>", "<h1>"):
            paragraph_open = True
            current_paragraph += 1

            continue
        elif l.lower() in ("</p>", "</hl>", "</h1>", "<p/>"):
            paragraph_open = False

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
        assert l.lower().startswith("<seg id=\"") and l.lower().endswith("</seg>"), l

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

        assert seg_id == current_sentence, f"{seg_id} vs {current_sentence}" # Some segments does not have correctly assigned the id...

        _trglang = trglang

        if not isinstance(translator, str) and translator and len(translator.groups()) > 0:
            #_trglang = f"{trglang}_{translator.group(1)}"
            translator = translator.group(1)

        if is_src_or_ref != src_or_ref:
            continue

        if doc_testsuite != "UNK": # testsuites often does not contain references, just source sentences
            continue

        result.append((setid, doc_origlang, _trglang, collection_name, doc_docid, doc_testsuite, current_document, current_paragraph, current_sentence, seg, seg_id, translator))

    assert not paragraph_open
    assert not document_open

    return result

def process_file(fn, src_or_ref):
    fd = open(fn, "rt")
    l = next(fd).strip()

    assert l == "<?xml version='1.0' encoding='utf-8'?>", l

    l = next(fd).strip()

    assert l.startswith("<dataset id="), l

    setid = p_setid.search(l)

    assert len(setid.groups()) == 1, f"{setid.group(0)} | {l}"

    setid = setid.group(1).replace('\t', ' ')
    collection_open = False
    collection_name = "UNK"
    data = []
    fake_collection = False
    results = {}

    for l in fd:
        l = l.strip()

        if l == "</collection>":
            assert collection_open
            assert collection_name not in results

            result = process_collection(data, collection_name, setid, src_or_ref)
            results[collection_name] = result
            collection_open = False
            collection_name = "UNK"
            data = []

            continue
        elif l == "</dataset>":
            if fake_collection:
                assert collection_name not in results

                result = process_collection(data, collection_name, setid, src_or_ref)
                results[collection_name] = result
                collection_open = False

            break
        elif not collection_open:
            assert not collection_open

            collection_open = True

            if l.startswith("<collection id="):
                assert not fake_collection, "A collection was found but data without collection was also provided!" # You lied!

                collection = p_collection.search(l)

                assert len(collection.groups()) == 1, f"{collection.group(0)} | {l}"

                collection_name = collection.group(1)

                continue
            else:
                fake_collection = True

        data.append(l)

    assert not collection_open

    fd.close()

    if src_or_ref == "ref":
        # merge translators
        placeholder_translator = "_PLACEHOLDER_TRANSLATOR_"
        placeholder_sentence = "_PLACEHOLDER_SENTENCE_"

        for collection_name, r in results.items():
            tmp_r = {}

            for sample in r:
                assert collection_name == sample[3]

                doc_docid = sample[4]
                doc_testsuite = sample[5]
                current_document = sample[6]
                current_paragraph = sample[7]
                current_sentence = sample[8]
                seg_id = sample[10]

                assert '@' not in str(doc_docid), doc_docid
                assert '@' not in str(doc_testsuite), doc_testsuite
                assert '@' not in str(current_document), current_document
                assert '@' not in str(current_paragraph), current_paragraph
                assert '@' not in str(current_sentence), current_sentence
                assert '@' not in str(seg_id), seg_id

                translator = sample[11]

                #k = f"{collection_name}@{doc_docid}@{doc_testsuite}@{current_document}@{current_paragraph}@{current_sentence}@{seg_id}"
                k = f"{collection_name}@{doc_docid}@{doc_testsuite}@{current_document}@{seg_id}"

                if k not in tmp_r:
                    tmp_r[k] = {}

                tmp_r[k][translator] = sample

            final_r = []

            for v in tmp_r.values():
                new_seg = {}

                for translator, sample in v.items():
                    assert translator not in new_seg

                    new_seg[translator] = sample[9] # seg

                    assert placeholder_translator not in translator
                    assert placeholder_translator not in sample[9]
                    assert placeholder_sentence not in translator
                    assert placeholder_sentence not in sample[9]

                new_seg = placeholder_sentence.join([f"{k}{placeholder_translator}{v}" for k, v in new_seg.items()])

                final_r.append(list(sample))

                final_r[-1][9] = new_seg # replace old seg
                final_r[-1] = tuple(final_r[-1])

            results[collection_name] = final_r # update

    return results

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

xml_fn = sys.argv[2]
src_result = process_file(xml_fn, "src")
ref_result = process_file(xml_fn, "ref")

assert src_result.keys() == ref_result.keys()

header = "xml_filename\tsetid\tsrc_lang\tref_lang\tcollection_name_similar_to_genre\tdocid\tnumber_doc\tnumber_paragraph\tnumber_sentence\tsrc_sentence\tref_sentence"
nheader = len(header.split('\t'))

print(header)

for collection_name in src_result.keys():
    assert len(src_result[collection_name]) == len(ref_result[collection_name]), f"{collection_name}: {len(src_result[collection_name])} vs {len(ref_result[collection_name])}"

    for src, ref in zip(src_result[collection_name], ref_result[collection_name]):
        #for i in (0, 1, 3, 4, 5, 6, 7, 8, 10):
        for i in (0, 1, 3, 4, 5, 6, 8, 10):
            check(src, ref, i, equal=True)
        for i in (2,):
            check(src, ref, i, equal=False)

        assert len(src) - 1 == len(ref) - 1 == nheader # - 1: segid, testsuite and translator not used (i.e., -3); xml fn (+1); both sentences are printed, but each variable contains just one (i.e., +1). -3 + 1 + 1 = - 1

        setid = src[0]
        doc_origlang = src[1]
        trglang = ref[2] # necessary to be ref!
        doc_genre = src[3]
        doc_docid = src[4]
        current_document = src[6]
        current_paragraph = src[7]
        current_sentence = src[8]
        src_seg = src[9]
        ref_seg = ref[9]

        assert doc_genre == collection_name

        if target_origlang != '-' and doc_origlang != target_origlang:
            continue
        if target_trglang != '-' and trglang != target_trglang:
            continue

        s = f"{xml_fn}\t{setid}\t{doc_origlang}\t{trglang}\t{doc_genre}\t{doc_docid}\t{current_document}\t{current_paragraph}\t{current_sentence}\t{src_seg}\t{ref_seg}"

        assert len(s.split('\t')) == nheader

        print(s)
