import logging

import csv

from typing import Dict, List, Iterable, Deque, Tuple, Any

from overrides import overrides
from collections import namedtuple, deque

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from transformers import BertTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import xml.etree.ElementTree as ElementTree

logger = logging.getLogger(__name__)

COL_INDEX = 0
COL_FORM = 1
COL_LEMMA = 2
COL_PLEMMA = 3
COL_POS = 4
COL_PPOS = 5
COL_FEAT = 6
COL_PFEAT = 7
COL_HEAD = 8
COL_PHEAD = 9
COL_DEPREL = 10
COL_PDEPREL = 11
COL_FILLPRED = 12
COL_PRED = 13
COL_APREDs = 14

Sentence = namedtuple("Sentence", ["words", "srltags"])
Word = namedtuple("Word", ["id", "form", "isverb", "verbframe", "head", "deprel", "upos"])
Span = namedtuple("Span", ["begin_i", "tag", "containsHead", "fromRoot"])
DepDerivation = namedtuple("DepDerivation", ["fromRoot", "tag"])


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]): # pylint: disable=invalid-name
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


@DatasetReader.register("conll2009_srl")
class Conll2010SrlReader(DatasetReader):
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
                 discard_normal_discontinuous_spans = True,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._discard_root_discontinuous_spans = discard_root_discontinuous_spans
        self._discard_normal_discontinuous_spans = discard_normal_discontinuous_spans

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False


    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

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



    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None,
                         no_arguments: bool = False) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        metadata_dict: Dict[str, Any] = {"no_arguments": no_arguments}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        fields['verb_indicator'] = verb_indicator

        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)



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

                    sentences[-1].words.append(Word(
                        id=int(row[COL_INDEX]),
                        form=row[COL_FORM],
                        isverb=(row[COL_FILLPRED] == "Y"),
                        verbframe=row[COL_PRED],
                        head=int(row[COL_HEAD]),
                        deprel=row[COL_DEPREL],
                        upos=row[COL_POS]
                    ))
                    # read all tag columns for this row at once, they are separated later
                    srltags: List[str] = row[COL_PRED + 1:len(row)]
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
