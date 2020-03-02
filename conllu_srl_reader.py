import logging

import csv

from typing import Dict, List, Iterable, Deque

from overrides import overrides
from collections import namedtuple, deque

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import xml.etree.ElementTree as ElementTree

logger = logging.getLogger(__name__)

COL_INDEX = 0
COL_FORM = 1
COL_LEMMA = 2
COL_UPOS = 3
COL_XPOS = 4
COL_FEATS = 5
COL_HEAD = 6
COL_DEPREL = 7
COL_ISVERB = 8
COL_VERBFRAME = 9

Sentence = namedtuple("Sentence", ["words", "srltags"])
Word = namedtuple("Word", ["id", "form", "isverb", "verbframe", "head", "deprel", "upos"])
Span = namedtuple("Span", ["begin_i", "tag", "containsHead", "fromRoot"])
DepDerivation = namedtuple("DepDerivation", ["fromRoot", "tag"])


class CustomSrlDatasetReader(DatasetReader):

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise NotImplementedError

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None,
                         no_arguments: bool = False) -> Instance:
        """
        Copied from semantic_role_labeling.py

        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        metadata_dict = {"words": [x.text for x in tokens],
                         "verb": verb,
                         "no_arguments": no_arguments}
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags
        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)


@DatasetReader.register("conllu_srl")
class ConlluSrlReader(CustomSrlDatasetReader):
    """
    Read ConLL-U file from https://github.com/System-T/UniversalPropositions/
    See more information at http://taiga.hucompute.org/project/wahed-allennlp/wiki/srl-dataformat

    Output is the same as SrlReader in AllenNLP: allennlp/data/dataset_readers/semantic_role_labeling.py.
    Docs copied here for convenience:
    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 discard_root_discontinuous_spans = True,
                 discard_normal_discontinuous_spans = True) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._discard_root_discontinuous_spans = discard_root_discontinuous_spans
        self._discard_normal_discontinuous_spans = discard_normal_discontinuous_spans

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        logger.info("Reading ConLL-U from %s", file_path)

        sentences = self.read_raw_sentences(file_path)

        for sentence in sentences:
            wordCount = len(sentence.words)
            for verb_i in range(len(sentence.srltags)):  # for each verb in the sentence
                tokens = [Token(word.form) for word in sentence.words]
                verb_one_hot = self.create_verb_onehot(sentence, verb_i, wordCount)
                tags = sentence.srltags[verb_i]
                biotags, discontinuous_span, root_discontinuous = self.sentence_to_bio(sentence, tags, verb_one_hot, wordCount)
                verb_has_no_arguments = tags.count('_') == len(tags)
                discard = (discontinuous_span and self._discard_normal_discontinuous_spans) \
                               or (root_discontinuous and self._discard_root_discontinuous_spans)
                if not discard or verb_has_no_arguments:
                    for i in range(len(verb_one_hot)):
                        if verb_one_hot[i] == 1:
                            biotags[i] = "B-V"
                    yield self.text_to_instance(tokens, verb_one_hot, biotags, verb_has_no_arguments)

    def read_raw_sentences(self, file_path):
        sentences: List[Sentence] = []
        with open(file_path, mode='r') as file:
            csvr = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
            # we disable quoting, because the default " can appear as punctuation character
            srlTagsInitDone = False  # have we already created a list for the tag columns?
            sentences.append(Sentence(words=[], srltags=[]))
            for row in csvr:
                # parse sentence
                try:
                    if len(row) == 0:  # new sentence
                        sentences.append(Sentence(words=[], srltags=[]))
                        srlTagsInitDone = False
                        continue
                    elif row[0][0] == '#':  # we skip comments
                        continue
                    elif '-' in row[COL_INDEX]:
                        continue  # this is a field that combines two words

                    if not row[COL_FORM]:
                        oldrow = row
                        row = [oldrow[COL_INDEX], oldrow[COL_FORM], ""]
                        row.extend(oldrow[2:len(oldrow) - 1])
                        # these lines are lacking one field containing the reduced form

                    sentences[-1].words.append(Word(
                        id=int(row[COL_INDEX]),
                        form=row[COL_FORM],
                        isverb=(row[COL_ISVERB] == "Y"),
                        verbframe=row[COL_VERBFRAME],
                        head=int(row[COL_HEAD]),
                        deprel=row[COL_DEPREL],
                        upos=row[COL_UPOS]
                    ))
                    # read all tag columns for this row at once, they are separated later
                    srltags: List[str] = row[COL_VERBFRAME + 1:len(row)]
                    if not srlTagsInitDone:
                        # this is the first word of the sentence, we have not yet created tag columns
                        srlTagsInitDone = True
                        for srltag in srltags:
                            sentences[-1].srltags.append([srltag])
                    else:
                        for i in range(len(srltags)):
                            sentences[-1].srltags[i].append(srltags[i])
                except:
                    # parse error :-( should not happen
                    print("error with line number", csvr.line_num, row)
                    raise
            # we might create one sentence too much, remove it
            if len(sentences[-1].words) == 0:
                sentences = sentences[:-1]
        return sentences

    def create_verb_onehot(self, sentence, verbi, wordCount):
        verbcount = 0
        findverbi = 0
        for i in range(wordCount):
            if sentence.words[i].isverb:
                findverbi = i
                verbcount += 1
                if verbcount == verbi + 1:
                    break
        verb_label = [1 if findverbi == i else 0 for i in range(wordCount)]
        return verb_label

    def sentence_to_bio(self, sentence, tags, verb_one_hot, wordCount):
        # if you want to use these, they need to be static over all runs of this function
        # discountCount = 0
        # discontCsv = open("./discontinuous_spans.csv", "w+")
        # rootCsv = open("root.csv", "w+")
        for root_alone in [False, True]:
            # first try with propagation of root tag to sentence
            tree_up, used_root_alone = self.dep_tree_to_arg_tags(sentence, tags, verb_one_hot, root_alone, wordCount)
            spans = self.create_spans(sentence, tags, tree_up, wordCount)
            biotags, discontinuous_span, disct_span_contains_root = self.spans_to_bio(sentence, spans, wordCount)

            # if one discontinuous span was derived from root, isolate root
            if not root_alone and discontinuous_span and disct_span_contains_root and\
                    not self._discard_root_discontinuous_spans:
                continue  # second try: let's try to isolate root

            return biotags, discontinuous_span, (discontinuous_span and disct_span_contains_root)

            # count and save discontinuousSpans
            # if discontinuous_span:
            #     discountCount += 1
            #     print(*tags, sep=";", file=discontCsv)
            #     print(*[w.head for w in sentence.words], sep=';', file=discontCsv)
            #     print(*tree_up, sep=";", file=discontCsv)
            #     print(*biotags, sep=";", file=discontCsv)
            #     print(*[w.form for w in sentence.words], sep=";", file=discontCsv)
            #     print("count", discountCount)
            #     print("", file=discontCsv)

            # print(used_root_alone, discontinuous_span)

    def dep_tree_to_arg_tags(self, sentence, tags, verb_label, root_alone, word_count):
        root = []
        tree_up = [DepDerivation(False, None) for _ in range(word_count)]
        for i in range(word_count):
            tree_up[i] = self.find_tag_head(tree_up, i, sentence.words, tags, verb_label, root_alone, root)
        root_alone_used = len(root) > 0
        return tree_up, root_alone_used

    def find_tag_head(self, tree_up, i, words: List[Word], argument_tags, verb_label, root_alone, root):
        if argument_tags[i] is not '_':
            return DepDerivation(words[i].head == 0, argument_tags[i])
        if verb_label[i] == 1:
            return DepDerivation(False, None)  # verbs are also O
        # we include punctuation, even fullstops, even if they are bad tagging

        hid = words[i].head-1
        if hid == -1:  # we are looking at the root word
            return DepDerivation(True, None) # other
        if tree_up[hid].tag is None:
            tree_up[hid] = self.find_tag_head(tree_up, hid, words, argument_tags, verb_label, root_alone, root)
        if root_alone and words[hid].head == 0:
            if tree_up[hid].tag is not None:
                root.append(True)
            return DepDerivation(True, None)  # don't derive tags from the root of the dependency tree
        return tree_up[hid]

    def create_spans(self, sentence, tags, tree_up, wordCount):
        # create spans
        spans = deque()
        spans.appendleft(Span(begin_i=-1, tag=None, containsHead=False, fromRoot=False))
        for i in range(wordCount):
            fromRoot, newTag = tree_up[i]
            if spans[0].tag != newTag:
                self.joinSpans(sentence, spans, i)
                spans.appendleft(Span(begin_i=i, tag=newTag, containsHead=False, fromRoot=False))
            # update containsHead property of current span -> we need to recreate the Span to write the property
            # we do this here because every new token in a span could contain the head of the span
            # same for fromRoot
            spans[0] = Span(begin_i=spans[0].begin_i,
                            tag=spans[0].tag,
                            containsHead=spans[0].containsHead or tags[i] == newTag,
                            fromRoot=spans[0].fromRoot or fromRoot)
        self.joinSpans(sentence, spans, wordCount)

        # fix first span
        if len(spans) >= 2 and spans[-2].begin_i == 0:
            spans.pop()  # remove the span from -1 to 0 (exclusive)
        else:
            # the span reaches from -1 to n, so set start to 0
            spans[-1] = Span(
                begin_i=0,
                tag=spans[-1].tag,
                containsHead=spans[-1].containsHead,
                fromRoot=spans[-1].fromRoot,
            )

        return spans

    def joinSpans(self, sentence, spans: Deque[Span], end_i):
        canJoin = len(spans) >= 3 \
                  and spans[2].tag == spans[0].tag \
                  and spans[0].begin_i - spans[1].begin_i == 1 \
                  and sentence.words[spans[1].begin_i].upos == "PUNCT"
        if canJoin and 0 <= spans[0].begin_i < end_i \
                and spans[0].tag is not None:
            # join the three spans
            containsHead = spans[0].containsHead or spans[2].containsHead
            fromRoot = spans[0].fromRoot or spans[2].fromRoot
            span0 = spans.popleft()  # current
            span1 = spans.popleft()  # punctuation
            span2 = spans.popleft()  # before punctuation
            # print("joining", span2, span1, span0)
            spans.appendleft(Span(begin_i=span2.begin_i,
                                  tag=span2.tag,
                                  containsHead=containsHead,
                                  fromRoot=fromRoot))

    def spans_to_bio(self, sentence, spans, word_count):
        # generate bio tags
        discontinuous_span = False
        discontinuous_span_contains_root = False
        biotags = ["O" for _ in range(word_count)]
        span = spans.pop()
        for i in range(word_count):
            if len(spans) > 0 and i >= spans[-1].begin_i:
                # check length because of last span
                span = spans.pop()

            if not span.containsHead and not span.tag is None:
                next_span_i = spans[0].begin_i if len(spans) > 0 else word_count
                if not (
                        next_span_i - span.begin_i == 1
                        and sentence.words[span.begin_i].deprel == "punct"
                ):
                    # we don't count punctuation that has a tag way outside of the actual span
                    discontinuous_span = True
                    discontinuous_span_contains_root = span.fromRoot or discontinuous_span_contains_root
                    # ^ there can be multiple discontinuous spans, if any of them contains root we want to know
                continue

            if span.tag is None:
                biotags[i] = "O"
            else:
                biotags[i] = ("B-" if i == span.begin_i else "I-") + span.tag
        return biotags, discontinuous_span, discontinuous_span_contains_root


@DatasetReader.register("salsa_srl")
class SalsaReader(CustomSrlDatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 discard_normal_discontinuous_spans: bool = True,
                 discard_subword_tokens: bool = True,
                 discard_multiple_verbs: bool = True,
                 ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._discard_normal_discontinuous_spans = discard_normal_discontinuous_spans
        self._discard_subword_tokens = discard_subword_tokens
        self._discard_multiple_verbs = discard_multiple_verbs

    def _read(self, file_path: str):
        event: str
        elem: ElementTree.Element
        for event, elem in ElementTree.iterparse(cached_path(file_path), events=("end",)):
            if elem.tag == "head":
                elem.clear()
                continue

            if elem.tag == "s":
                euspframes: ElementTree.Element = elem.find("sem/usp/uspframes")
                euspfes: ElementTree.Element = elem.find("sem/usp/uspfes")
                if len(list(euspframes)) > 0 or len(list(euspfes)) > 0:
                    elem.clear()
                    continue  # discard unspecified

                sentence_id = elem.attrib["id"]
                tree_by_id = {}
                framei = 0

                try:
                    eterminals = elem.find("graph/terminals").findall("t")
                    tokeni = 0
                    for eterminal in eterminals:
                        tree_by_id[eterminal.attrib["id"]] = {
                            "token": eterminal.attrib["word"],
                            "tokeni": tokeni,
                            "pos": eterminal.attrib["pos"]
                        }
                        tokeni += 1

                    esplitwords = elem.find("sem/splitwords")
                    if esplitwords is not None:
                        esplitwords = esplitwords.findall("splitword")
                        if len(esplitwords) > 0 and self._discard_subword_tokens:
                            elem.clear()
                            continue  # discard this sentence
                        for esplitword in esplitwords:
                            idref = esplitword.attrib["idref"]
                            tokeni = tree_by_id[idref]["tokeni"]  # this will definitely refer to a token
                            part_ids = []
                            for epart in esplitword.findall("part"):
                                tokeni += 0.01 # space for 99 word parts
                                id = epart.attrib["id"]
                                tree_by_id[id] = {
                                    "token": epart.attrib["word"],
                                    "tokeni": tokeni,
                                    "pos": tree_by_id[idref]["pos"]
                                }
                                part_ids.append(id)
                            tree_by_id[idref] = {"children": [tree_by_id[id] for id in part_ids]}

                    enonterminals = elem.find("graph/nonterminals").findall("nt")
                    enonterminal: ElementTree.Element
                    for enonterminal in enonterminals:
                        tree_by_id[enonterminal.attrib["id"]] = {
                            "children":[tree_by_id[eedge.attrib["idref"]] for eedge in enonterminal.findall("edge")]
                        }

                    for eframe in elem.findall("sem/frames/frame"):

                        if not eframe.attrib["id"][0].islower():
                            # print("discarding noun frame", eframe.attrib["id"], sentence_id)
                            continue  # discard noun frames

                        # print("keeping verb frame", eframe.attrib["id"], sentence_id)

                        etargets = eframe.findall("target")
                        if (0 == len(etargets) or (len(etargets) > 1) or any([len(etarget) > 1 for etarget in etargets]))\
                                and self._discard_multiple_verbs:
                            # only catches obvious cases of multiple verbs, we do a thorough check later
                            continue  # discard

                        tag_overlap = []
                        for etarget in etargets:
                            for efenode in etarget.findall("fenode"):
                                tag_overlap.append(
                                    self.tag_terminals(tree_by_id[efenode.attrib["idref"]], framei, "V")
                                )
                        for efe in eframe.findall("fe"):
                            tag = efe.attrib["name"]
                            for efenode in efe.findall("fenode"):
                                tag_overlap.append(
                                    self.tag_terminals(tree_by_id[efenode.attrib["idref"]], framei, tag)
                                )
                        if any(tag_overlap):
                            # we cannot process overlapping frames
                            continue  # discard tag overlap

                        framei += 1

                    if framei == 0:
                        # no frame found in this sentence
                        elem.clear()
                        continue

                except KeyError as err:
                    key: str = err.args[0]
                    if not key.startswith(sentence_id):
                        # print(key, "not part of sentence", sentence_id)
                        # we do not support annotations over sentence boundaries
                        continue
                    else:
                        print(key, "not found although part of sentence")

                discard_frames = []
                tokenobjs = list(sorted(filter(lambda x: "tokeni" in x, tree_by_id.values()), key=lambda x: x["tokeni"]))
                for frame in range(framei):
                    lasttag = None
                    for i in range(len(tokenobjs)):
                        tokenobj = tokenobjs[i]
                        if frame not in tokenobj:
                            tokenobj[frame] = "O"
                            continue

                        if i-2 >= 0:
                            # join spans over comma borders
                            if tokenobjs[i-2][frame][2:] == tokenobj[frame] \
                                    and tokenobjs[i-1]["pos"][0] == "$" \
                                    and tokenobjs[i-1][frame] == "O":
                                tokenobjs[i-1][frame] = "I-" + tokenobj[frame]

                        if i-1 >= 0 and tokenobjs[i-1][frame][2:] == tokenobj[frame]:
                            if tokenobj[frame] == "V" and self._discard_multiple_verbs:
                                # verb consists of multiple words
                                # we can only detect this here because it would be complicated searching
                                # through the tree structure
                                discard_frames.append(frame)
                                break
                            tokenobj[frame] = "I-" + tokenobj[frame]
                        else:
                            tokenobj[frame] = "B-" + tokenobj[frame]

                for frame in range(framei):
                    if frame in discard_frames:
                        continue
                    containsVerb = False
                    spanBeginnings = []
                    for tokenobj in tokenobjs:
                        if self._discard_multiple_verbs:
                            if tokenobj[frame][2:] == "V":
                                if containsVerb:
                                    discard_frames.append(frame)
                                    # print("discarding because multiple B-V", sentence_id, frame, "of", framei)
                                    break
                                containsVerb = True
                        if self._discard_normal_discontinuous_spans:
                            tag = tokenobj[frame]
                            if tag[:1] == "B":
                                if tag == "B-V":
                                    continue  # B-V is controlled by another parameter
                                if tag in spanBeginnings:
                                    discard_frames.append(frame)
                                    # print("discard because discontinuous span", sentence_id, frame, "of", framei)
                                    break
                                spanBeginnings.append(tag)

                for frame in range(framei):
                    if frame in discard_frames:
                        continue
                    tokens = [Token(t["token"]) for t in tokenobjs]
                    tags = [t[frame] for t in tokenobjs]
                    verbs = [1 if t[frame][2:] == "V" else 0 for t in tokenobjs]
                    yield self.text_to_instance(tokens, verbs, tags)

                elem.clear()

    def tag_terminals(self, tree, index, tag):
        if "tokeni" in tree:
            if index in tree:
                return True
            tree[index] = tag
            return False
        elif "children" in tree:
            tag_overlap = []
            for child in tree["children"]:
                tag_overlap.append(self.tag_terminals(child, index, tag))
            return any(tag_overlap)
        else:
            print(tree, index, tag)
            raise Exception("Tree node")
