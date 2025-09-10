import numpy as np
from typing import Any
import itertools
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from ..tokenizer.tokenizer import create_custom_tokenizer, wrap_tokenizer

class XAYBZ:
    def __init__(self,
                 num_ab_categories=2,
                 ab_category_size=3,

                 num_x_categories=0,
                 x_category_size=0,

                 num_y_categories=0,
                 y_category_size=3,

                 num_z_categories=0,
                 z_category_size=0,

                 min_x_per_sentence=0,
                 max_x_per_sentence=0,
                 min_y_per_sentence=1,
                 max_y_per_sentence=1,
                 min_z_per_sentence=0,
                 max_z_per_sentence=0,
                 num_omitted_ab_pairs=1,

                 document_organization_rule='all_pairs',  # 'all_pairs', 'one_pair_each_category', 'single_sentence', 'single_category'
                 document_repetitions=1,  # 1: No repetitions, 2: Repeat each group's docs reps times before moving to next group
                 document_sequence_rule='massed',  # 'massed', 'interleaved', 'random'

                 sentence_repetitions_per_document=0,
                 # 0: No repetitions, 1: Repeat each sentence reps times, 2: Repeat each sentence reps times for each subcategory
                 sentence_repetition_type='same_subcategory',
                 sentence_sequence_rule='massed',  # 'massed', 'interleaved', 'random'

                 include_punctuation=True,
                 include_pad=True,
                 include_unknown=True,
                 include_bos=False,
                 include_eos=False,

                 custom = None,

                 random_seed=None
                 ):
        self.num_ab_categories = num_ab_categories
        self.ab_category_size = ab_category_size
        self.num_x_categories = num_x_categories
        self.x_category_size = x_category_size
        self.num_y_categories = num_y_categories
        self.y_category_size = y_category_size
        self.num_z_categories = num_z_categories
        self.z_category_size = z_category_size
        self.min_x_per_sentence = min_x_per_sentence
        self.max_x_per_sentence = max_x_per_sentence
        self.min_y_per_sentence = min_y_per_sentence
        self.max_y_per_sentence = max_y_per_sentence
        self.min_z_per_sentence = min_z_per_sentence
        self.max_z_per_sentence = max_z_per_sentence
        self.num_omitted_ab_pairs = num_omitted_ab_pairs
        self.document_organization_rule = document_organization_rule
        self.document_repetitions = document_repetitions
        self.document_sequence_rule = document_sequence_rule
        self.sentence_repetitions_per_document = sentence_repetitions_per_document
        self.sentence_repetition_type = sentence_repetition_type
        self.sentence_sequence_rule = sentence_sequence_rule
        self.include_pad = include_pad
        self.include_unknown = include_unknown
        self.include_bos = include_bos
        self.include_eos = include_eos
        self.include_punctuation = include_punctuation
        self.custom = custom
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)
        self.create_dataset_name()
        self.check_parameters()
        self.create_dataset()
        print(self)
    
    def create_dataset_name(self):
        parts = [
            "xAyBz",
            self.num_ab_categories,
            self.ab_category_size,
            self.num_omitted_ab_pairs,
            self.num_x_categories,
            self.x_category_size,
            self.min_x_per_sentence,
            self.max_x_per_sentence,
            self.num_y_categories,
            self.y_category_size,
            self.min_y_per_sentence,
            self.max_y_per_sentence,
            self.num_z_categories,
            self.z_category_size,
            self.min_z_per_sentence,
            self.max_z_per_sentence,
            self.document_organization_rule,
            self.document_repetitions,
            self.document_sequence_rule,
            self.sentence_repetitions_per_document,
            self.sentence_sequence_rule,
            self.random_seed
        ]
        self.dataset_name = "_".join(map(str, parts))
        print(f"\nCreating dataset {self.dataset_name}")

    def __repr__(self):
        lines = [
            f"\n{self.dataset_name}\n",
            f"    Vocab Size: {self.generated_vocabulary_size}",
            f"    Special Tokens: [{','.join(self.special_tokens)}]",
            f"    Punctuation: [.]", 
            f"    x Category: [{','.join(self.x_list)}]",
            f"    y Category: [{','.join(self.y_list)}]",
            f"    z Category: [{','.join(self.z_list)}]",
        ]
        for category, (a_words, b_words) in self.ab_category_dict.items():
            lines.append(f"    A{category}: [{','.join(a_words)}]")
            lines.append(f"    B{category}: [{','.join(b_words)}]")
        lines.append(f"    Vocabulary: [{','.join(self.generated_vocab_list)}]",)
        
        lines.append("    Documents:")
        for i, doc in enumerate(self.generated_document_list):
            group = self.generated_document_group_list[i]
            lines.append(f"        Document:{i} Group:{group} Len:{len(doc)}")
            for sent in doc:
                lines.append(f"            [{','.join(sent)}]")

        return "\n".join(lines) + "\n"

    def save_as_jsonl(self, out_dir: str):
        dataset_name = self.dataset_name
        root = Path(out_dir) / dataset_name
        root.mkdir(parents=True, exist_ok=True)
        for i, doc in enumerate(self.generated_document_list):
            group = self.generated_document_group_list[i]
            labels = self.generated_document_labels_list[i]
            assert len(doc) == len(labels), "Document and labels must have the same length"
            out_path = root / f"Document:{i} Group:{group} Len:{len(doc)}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for sent_id, (sentence, label) in enumerate(zip(doc, labels)):
                    row = {
                        "doc_id": i,
                        "sent_id": sent_id,
                        "group": group,
                        "text": " ".join(sentence),
                        "tokens": sentence,
                        "label": label,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_xaybz(self):
        for i, doc in enumerate(self.generated_document_list):
            group = self.generated_document_group_list[i]
            labels = self.generated_document_labels_list[i]
            assert len(doc) == len(labels), "Document and labels must have the same length"
            for sent_id, (sentence, label) in enumerate(zip(doc, labels)):
                row = {
                    "doc_id": i,
                    "sent_id": sent_id,
                    "group": group,
                    "text": " ".join(sentence),
                    "tokens": sentence,
                    "label": label,
                }
                yield row

    def create_dataset(self):
        self.create_vocabulary()
        self.build_tokenizer()
        self.create_vocab_pair_list()
        if self.custom in ["and", "xnor"]:
            self.create_logic_document_list()
        else:
            self.create_document_list()

    def check_parameters(self):
        def check(cond, msg):
            if not cond:
                raise ValueError(msg)

        # AB categories
        check(self.num_ab_categories >= 1,
            "num_ab_categories must be >= 1")
        check(self.ab_category_size >= 2,
            "ab_category_size must be >= 2")
        check(0 <= self.num_omitted_ab_pairs < self.ab_category_size,
            "num_omitted_ab_pairs must be >= 0 and less than ab_category_size")

        # X
        check(self.x_category_size >= 1 or self.num_x_categories == 0,
            "x_category_size must be >= 1 if num_x_categories > 0")
        check(0 <= self.min_x_per_sentence <= self.max_x_per_sentence,
            "Require 0 <= min_x_per_sentence <= max_x_per_sentence")

        # Y
        check(self.y_category_size >= 1 or self.num_y_categories == 0,
            "y_category_size must be >= 1 if num_y_categories > 0")
        check(0 <= self.min_y_per_sentence <= self.max_y_per_sentence,
            "Require 0 <= min_y_per_sentence <= max_y_per_sentence")

        # Z
        check(self.z_category_size >= 1 or self.num_z_categories == 0,
            "z_category_size must be >= 1 if num_z_categories > 0")
        check(0 <= self.min_z_per_sentence <= self.max_z_per_sentence,
            "Require 0 <= min_z_per_sentence <= max_z_per_sentence")

        # Document-level
        check(self.document_repetitions >= 1,
            "document_repetitions must be >= 1")
        check(self.document_organization_rule in
            {"all_pairs", "one_pair_each_category", "single_sentence", "single_category"},
            f"Unrecognized document_organization_rule {self.document_organization_rule}")
        check(self.document_sequence_rule in
            {"massed", "interleaved", "random"},
            f"Unrecognized document_sequence_rule {self.document_sequence_rule}")
    @staticmethod
    def _parse_cat_idx(word: str):
        if word == ".":
            return None, None
        if "_" not in word:
            # (Optional) legacy fallback like "A32" -> cat=3, idx=2
            s = word
            if s and s[0] in "ABxyz":
                s = s[1:]
            if s.isdigit() and len(s) >= 2 and word[0] in "AB":
                return int(s[:-1]), int(s[-1])
            return None, None

        left, right = word.split("_", 1)
        cat_str = left[1:] if left and left[0] in "ABxyz" else ""
        cat = int(cat_str) if cat_str.isdigit() else None
        idx = int(right) if right.isdigit() else None
        return cat, idx

    def create_vocabulary(self):
        # --- reset containers ---
        self.generated_vocab_list = []
        self.generated_vocab_index_dict = {}
        self.generated_index_vocab_dict = {}
        self.generated_vocabulary_size = 0

        self.ab_category_dict = {}   # int -> (A_words, B_words)
        self.x_category_dict = {}    # int -> [x words]
        self.y_category_dict = {}    # int -> [y words]
        self.z_category_dict = {}    # int -> [z words]

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.special_tokens = []
        self.special_tokens_dict = {}

        self.x_indices = []
        self.y_indices = []
        self.z_indices = []
        self.dot_index = None
        self.pad_index = None
        self.unk_index = None
        self.bos_index = None
        self.eos_index = None
        self.a_indices_by_cat = {}
        self.b_indices_by_cat = {}

        self.indices_by_category = defaultdict(list)

        self.vocab_target_category_list_dict = {}

        def add_words(words):
            for w in words:
                if w not in self.generated_vocab_index_dict:
                    idx = len(self.generated_vocab_list)
                    self.generated_vocab_index_dict[w] = idx
                    self.generated_index_vocab_dict[idx] = w
                    self.generated_vocab_list.append(w)
                    self.indices_by_category[w.split("_")[0]].append(idx)

        def mk_members(size: int, prefix: str) -> list[str]:
            # e.g., mk_members(3, "A1_") -> ["A1_1","A1_2","A1_3"]
            return [f"{prefix}{i}" for i in range(1, size + 1)] if size > 0 else []
        
        if self.include_pad:
            add_words(["<pad>"])
            self.pad_index = self.generated_vocab_index_dict["<pad>"]
            self.special_tokens.append("<pad>")
            self.special_tokens_dict["<pad>"] = "<pad>"
        else:
            self.special_tokens_dict["<pad>"] = None
        if self.include_unknown:
            add_words(["<unk>"])
            self.unk_index = self.generated_vocab_index_dict["<unk>"]
            self.special_tokens.append("<unk>")
            self.special_tokens_dict["<unk>"] = "<unk>"
        else:
            self.special_tokens_dict["<unk>"] = None
        if self.include_bos:
            add_words(["<bos>"])
            self.bos_index = self.generated_vocab_index_dict["<bos>"]
            self.special_tokens.append("<bos>")
            self.special_tokens_dict["<bos>"] = "<bos>"
        else:
            self.special_tokens_dict["<bos>"] = None
        if self.include_eos:
            add_words(["<eos>"])
            self.eos_index = self.generated_vocab_index_dict["<eos>"]
            self.special_tokens.append("<eos>")
            self.special_tokens_dict["<eos>"] = "<eos>"
        else:
            self.special_tokens_dict["<eos>"] = None
        if self.include_punctuation:
            add_words(["."])
            self.dot_index = self.generated_vocab_index_dict["."]

        for cat in range(1, self.num_ab_categories + 1):
            a_prefix = f"A{cat}_"
            b_prefix = f"B{cat}_"
            a_words = mk_members(self.ab_category_size, a_prefix)
            b_words = mk_members(self.ab_category_size, b_prefix)
            add_words(a_words)
            add_words(b_words)
            self.a_indices_by_cat[cat] = [self.generated_vocab_index_dict[a] for a in a_words]
            self.b_indices_by_cat[cat] = [self.generated_vocab_index_dict[b] for b in b_words]
            self.ab_category_dict[cat] = (a_words, b_words)
            for t in a_words + b_words:
                self.vocab_target_category_list_dict[t] = cat

        # --- X categories (allow pooled fallback like Y/Z for consistency) ---
        if self.num_x_categories > 0:
            for cat in range(1, self.num_x_categories + 1):
                    xs = mk_members(self.x_category_size, f"x{cat}_")
                    add_words(xs)
                    self.x_indices.extend([self.generated_vocab_index_dict[x] for x in xs])
                    self.x_category_dict[cat] = xs
                    self.x_list.extend(xs)
        elif self.x_category_size > 0:
            # pooled X (no categories)
            self.x_list = mk_members(self.x_category_size, "x")
            add_words(self.x_list)
            self.x_indices = [self.generated_vocab_index_dict[x] for x in self.x_list]
        
        # --- Y categories ---
        if self.num_y_categories > 0:
            for cat in range(1, self.num_y_categories + 1):
                ys = mk_members(self.y_category_size, f"y{cat}_")
                add_words(ys)
                self.y_indices.extend([self.generated_vocab_index_dict[y] for y in ys])
                self.y_category_dict[cat] = ys
                self.y_list.extend(ys)
        elif self.y_category_size > 0:
            # pooled Y (no categories)
            self.y_list = mk_members(self.y_category_size, "y")
            add_words(self.y_list)
            self.y_indices = [self.generated_vocab_index_dict[y] for y in self.y_list]

        # --- Z categories ---
        if self.num_z_categories > 0:
            for cat in range(1, self.num_z_categories + 1):
                zs = mk_members(self.z_category_size, f"z{cat}_")
                add_words(zs)
                self.z_indices.extend([self.generated_vocab_index_dict[z] for z in zs])
                self.z_category_dict[cat] = zs
                self.z_list.extend(zs)
        elif self.z_category_size > 0:
            # pooled Z (no categories)
            self.z_list = mk_members(self.z_category_size, "z")
            add_words(self.z_list)
            self.z_indices = [self.generated_vocab_index_dict[z] for z in self.z_list]
    
        self.generated_vocabulary_size = len(self.generated_vocab_list)
        self.build_vocab_category_maps()
    
    def build_vocab_category_maps(self):
        vocab_list = self.generated_vocab_list

        self.vocab_to_category = {}
        self.vocab_to_subcategory = {} 

        for vocab in vocab_list:
            cat, idx = XAYBZ._parse_cat_idx(vocab)
            self.vocab_to_category[vocab] = cat
            self.vocab_to_subcategory[vocab] = idx

        self.index_to_category = {}
        self.index_to_subcategory = {}
        for idx, vocab in self.generated_index_vocab_dict.items():
            c, s = self.vocab_to_category[vocab], self.vocab_to_subcategory[vocab]
            self.index_to_category[idx] = c
            self.index_to_subcategory[idx] = s

    def build_tokenizer(self):
        self.tokenizer = create_custom_tokenizer(
            vocab=self.generated_vocab_index_dict
        )
        self.tokenizer = wrap_tokenizer(self.tokenizer)

    def create_vocab_pair_list(self):
        # validate
        if not (0 <= self.num_omitted_ab_pairs < self.ab_category_size):
            raise ValueError(
                "num_omitted_ab_pairs must be in [0, ab_category_size-1]"
            )

        self.included_ab_pair_list = []          # flat (Ai, Bj)
        self.omitted_ab_pair_list = []           # flat (Ai, Bj)
        self.included_pairs_by_cat = {}          # cat -> list[(Ai, Bj)]
        self.omitted_pairs_by_cat = {}           # cat -> list[(Ai, Bj)]
        self.legal_ab_matrix_by_cat = {}         # cat -> (size x size) int array

        for cat, (a_set, b_set) in self.ab_category_dict.items():
            legal = np.zeros((self.ab_category_size, self.ab_category_size), dtype=int)
            inc, om = [], []
            for i in range(self.ab_category_size):
                for j in range(self.ab_category_size):
                    # omit k circular diagonals nearest the main diagonal
                    omit_index = abs(i - j) % self.ab_category_size
                    if omit_index < self.num_omitted_ab_pairs:
                        om.append((a_set[i], b_set[j]))
                    else:
                        inc.append((a_set[i], b_set[j]))
                        legal[i, j] = 1

            self.included_pairs_by_cat[cat] = inc
            self.omitted_pairs_by_cat[cat] = om
            self.legal_ab_matrix_by_cat[cat] = legal

            # maintain the existing flat lists for backwards compatibility
            self.included_ab_pair_list.extend(inc)
            self.omitted_ab_pair_list.extend(om)

    def create_sentence(self, ab_pair, current_x_list=None, current_y_list=None, current_z_list=None):
        # local RNG
        rng = self.rng

        # sample counts
        num_x = rng.integers(self.min_x_per_sentence, self.max_x_per_sentence + 1)
        num_y = rng.integers(self.min_y_per_sentence, self.max_y_per_sentence + 1)
        num_z = rng.integers(self.min_z_per_sentence, self.max_z_per_sentence + 1)

        # helpers that handle empty inventories gracefully
        def pick_many(pool, n):
            if not pool or n <= 0:
                return []
            # with replacement; change to without if needed
            return [pool[int(rng.integers(0, len(pool)))] for _ in range(n)]

        Xs = current_x_list if current_x_list is not None else pick_many(self.x_list, num_x)
        Ys = current_y_list if current_y_list is not None else pick_many(self.y_list, num_y)
        Zs = current_z_list if current_z_list is not None else pick_many(self.z_list, num_z)

        # base structure
        sentence = []
        sentence.extend(Xs)
        sentence.append(ab_pair[0])
        sentence.extend(Ys)
        sentence.append(ab_pair[1])
        sentence.extend(Zs)

        # punctuation
        if self.include_punctuation:
            sentence.append(".")

        return sentence        

    def create_labels_for_sentence(self, a_vocab: str, b_vocab: str, y_cat: int | None):
        generated_vocab_size = len(self.generated_vocab_list)
        labels = ["Other"] * generated_vocab_size
        to_idx = self.generated_vocab_index_dict
        a_idx = to_idx[a_vocab]
        b_idx = to_idx[b_vocab]
        a_cat = self.vocab_to_category[a_vocab]
        b_cat = self.vocab_to_category[b_vocab]
        a_inst = self.vocab_to_subcategory[a_vocab]  # 1-based
        b_inst = self.vocab_to_subcategory[b_vocab]

        
        for i in self.x_indices: labels[i] = "x"
        for i in self.y_indices: labels[i] = "y"
        for i in self.z_indices: labels[i] = "z"
        if self.dot_index is not None:
            labels[self.dot_index] = "."

        A_by_cat = self.a_indices_by_cat
        B_by_cat = self.b_indices_by_cat

        if self.custom in ["and", "xnor"]:
            a_true = (a_cat == 2)
            y_true = (y_cat == 2)
            if self.custom == "and":
                intended_b_cat = 2 if (a_true and y_true) else 1
            else:  # xnor
                intended_b_cat = 2 if (a_true == y_true) else 1

            for cat, idxs in B_by_cat.items():
                lab = "B_Legal" if cat == intended_b_cat else "B_Illegal"
                for i in idxs:
                    labels[i] = lab
            for cat, idxs in A_by_cat.items():
                lab = "A_Legal" if cat == intended_b_cat else "A_Illegal"
                for i in idxs:
                    labels[i] = lab
        else:
            for cat, idxs in A_by_cat.items():
                if cat != b_cat:
                    for i in idxs:
                        labels[i] = "A_Illegal"
                else:
                    for i in idxs:
                        t_inst = self.index_to_subcategory[i]  # 1-based
                        legal = self.legal_ab_matrix_by_cat[cat][t_inst - 1, b_inst - 1]
                        labels[i] = "A_Legal" if legal == 1 else "A_Omitted"

            for cat, idxs in B_by_cat.items():
                if cat != a_cat:
                    for i in idxs:
                        labels[i] = "B_Illegal"
                else:
                    for i in idxs:
                        t_inst = self.index_to_subcategory[i]  # 1-based
                        legal = self.legal_ab_matrix_by_cat[cat][t_inst - 1, a_inst - 1]
                        labels[i] = "B_Legal" if legal == 1 else "B_Omitted"
        return labels

    def assign_legality_to_sentence(self, sentence: list[str]):
        a_tok = next(t for t in sentence if t and t[0] == "A")
        b_tok = next(t for t in sentence if t and t[0] == "B")
        y_tok = next((t for t in sentence if t and t[0] == "y"))
        y_cat = self.vocab_to_category[y_tok]
        labels = self.create_labels_for_sentence(a_tok, b_tok, y_cat)
        return labels
    
    def create_document_list(self):
        rng = self.rng

        by_category = self.included_pairs_by_cat
        categories = sorted(by_category.keys())

        group_docs = {}   # group_label -> list[list[sentence]]
        group_order = []  # stable order of groups

        org = self.document_organization_rule
        if org == "all_pairs":
            group = "all"
            doc = []
            for c in categories:
                doc.extend(by_category.get(c, []))
            group_docs[group] = [doc]
            group_order = [group]

        elif org == "one_pair_each_category":
            max_len = max(len(by_category.get(c, [])) for c in categories)
            docs = []
            for r in range(max_len):
                doc_r = []
                for c in categories:
                    pairs_c = by_category.get(c, [])
                    if r < len(pairs_c):
                        doc_r.append(pairs_c[r])
                if doc_r:
                    docs.append(doc_r)
            group = "one_per_cat"
            group_docs[group] = docs
            group_order = [group]

        elif org == "single_sentence":
            docs = [[pair] for pair in self.included_ab_pair_list]
            group_docs["single"] = docs
            group_order = ["single"]
        

        elif org == "single_category":
            # One document per category with all its pairs
            for c in categories:
                group = f"cat{c}"
                group_docs[group] = [list(by_category.get(c, []))]
                group_order.append(group)

        else:
            raise ValueError(f"Unrecognized document_organization_rule: {org}")

        seq = self.document_sequence_rule
        reps = self.document_repetitions

        full_document_templates = []   # list[list[sentence]]
        full_document_group_labels = []

        if seq == "massed":
            # Repeat each group's docs reps times before moving to next group
            for g in group_order:
                docs = group_docs[g]
                for _ in range(reps):
                    for d in docs:
                        full_document_templates.append(list(d))  # shallow copy
                        full_document_group_labels.append(g)
        
        elif seq == "interleaved":
            # Round-robin across groups for each repetition
            for _ in range(reps):
                # Interleave by the longest group length
                max_docs = max(len(group_docs[g]) for g in group_order)
                for i in range(max_docs):
                    for g in group_order:
                        docs = group_docs[g]
                        if i < len(docs):
                            full_document_templates.append(list(docs[i]))
                            full_document_group_labels.append(g)

        elif seq == "random":
            # Build the massed list, then shuffle documents & labels together with self.rng
            staged = []
            for g in group_order:
                docs = group_docs[g]
                for _ in range(reps):
                    for d in docs:
                        staged.append((list(d), g))
            # shuffle deterministically
            idx = list(range(len(staged)))
            rng.shuffle(idx)
            for k in idx:
                d, g = staged[k]
                full_document_templates.append(d)
                full_document_group_labels.append(g)

        else:
            raise ValueError(f"Unrecognized document_sequence_rule {seq}")
        
        y_lists = []
        for ln in range(self.min_y_per_sentence, self.max_y_per_sentence + 1):
            if ln == 0:
                y_lists.append(()) 
            else:
                y_lists += list(itertools.product(self.y_list, repeat=ln))

            self.generated_document_list = []
            self.generated_document_labels_list = []
            self.generated_document_group_list = []

            reps_per_doc = self.sentence_repetitions_per_document
            rep_type = self.sentence_repetition_type
            sent_seq = self.sentence_sequence_rule

            for doc_tmpl, group_label in zip(full_document_templates, full_document_group_labels):
                generated_sentences = []
                generated_labels = []
                if reps_per_doc == 0:
                    if sent_seq in ("massed", "random"):
                        for (Ai, Bj) in doc_tmpl:
                            for ys in y_lists:
                                generated_sentences.append(self.create_sentence((Ai, Bj), current_y_list=ys))
                                generated_labels.append(self.assign_legality_to_sentence(generated_sentences[-1]))
                        if sent_seq == "random":
                            shuffled_idx = self.rng.permutation(len(generated_sentences))
                            generated_sentences = [generated_sentences[i] for i in shuffled_idx]
                            generated_labels = [generated_labels[i] for i in shuffled_idx]

                    elif sent_seq == "interleaved":
                        for ys in y_lists:
                            for (Ai, Bj) in doc_tmpl:
                                generated_sentences.append(self.create_sentence((Ai, Bj), current_y_list=ys))
                                generated_labels.append(self.assign_legality_to_sentence(generated_sentences[-1]))

                    else:
                        raise ValueError(f"unrecognized sentence_sequence_rule={sent_seq}")
                else:
                    # repetitions > 0
                    if sent_seq in ("massed", "random"):
                        for (Ai, Bj) in doc_tmpl:
                            for i, ys in enumerate(y_lists):
                                if rep_type == "same_subcategory":
                                    # Repeat only when A, B, Y[0] share category
                                    try:
                                        catA = self.vocab_to_category[Ai]
                                        catB = self.vocab_to_category[Bj]
                                        catY = self.vocab_to_category[ys[0]] if len(ys) > 0 else None
                                        do_repeat = (catY is not None) and (catA == catB == catY)
                                    except Exception:
                                        do_repeat = False
                                    n = reps_per_doc if do_repeat else 1
                                    for _ in range(n):
                                        generated_sentences.append(self.create_sentence((Ai, Bj), current_y_list=ys))
                                        generated_labels.append(self.assign_legality_to_sentence(generated_sentences[-1]))
                                else:
                                    # generic repetition without Y constraint
                                    for _ in range(reps_per_doc):
                                        generated_sentences.append(self.create_sentence((Ai, Bj), current_y_list=None))
                                        generated_labels.append(self.assign_legality_to_sentence(generated_sentences[-1]))
                        if sent_seq == "random":
                            shuffled_idx = self.rng.permutation(len(generated_sentences))
                            generated_sentences = [generated_sentences[i] for i in shuffled_idx]
                            generated_labels = [generated_labels[i] for i in shuffled_idx]

                    elif sent_seq == "interleaved":
                        for _ in range(reps_per_doc):
                            for (Ai, Bj) in doc_tmpl:
                                generated_sentences.append(self.create_sentence((Ai, Bj), current_y_list=None))
                                generated_labels.append(self.assign_legality_to_sentence(generated_sentences[-1]))
                    else:
                        raise ValueError(f"unrecognized sentence_sequence_rule={sent_seq}")
                self.generated_document_list.append(generated_sentences)
                self.generated_document_labels_list.append(generated_labels)
                self.generated_document_group_list.append(group_label)
            self.generated_num_documents = len(self.generated_document_list)

    def create_logic_document_list(self):
            ab_nc = len(self.ab_category_dict)
            y_nc  = len(self.y_category_dict)

            if ab_nc != 2:
                raise ValueError(f"custom '{self.custom}' requires exactly 2 A/B categories; got {ab_nc}")
            if y_nc != 2:
                raise ValueError(f"custom '{self.custom}' requires exactly 2 Y categories; got {y_nc}")

            rng = self.rng

            def logic_fn(rule: str, a_cat: int, y_cat: int) -> int:
                a_true = (a_cat == 2)
                y_true = (y_cat == 2)
                if rule == "and":
                    return 2 if (a_true and y_true) else 1
                elif rule == "xnor":
                    return 2 if (a_true == y_true) else 1
                else:
                    raise ValueError(f"Unsupported custom rule: {rule}")

            b_words_by_cat = {int(cat): b_words 
                              for cat, (_a_words, b_words) in self.ab_category_dict.items() }
            def _iter_y():
                for ycat, ys in self.y_category_dict.items():
                    ycat = int(ycat)
                    for yt in ys:
                        yield ycat, yt

            self.generated_document_list = []
            self.generated_document_labels_list = []
            self.generated_document_group_list = []

            sentences = []
            labels = []
            group_label = self.custom

            for a_cat, (a_words, _unused_b_words) in self.ab_category_dict.items():
                a_cat = int(a_cat)
                for a_tok in a_words:
                    for y_cat, y_tok in _iter_y():
                        b_cat = logic_fn(self.custom, a_cat, y_cat)
                        for b_tok in b_words_by_cat.get(b_cat, []):
                            sentence = self.create_sentence((a_tok, b_tok), current_y_list=[y_tok])
                            sentences.append(sentence)
                            labels.append(self.assign_legality_to_sentence(sentence))

            if self.sentence_sequence_rule == "random":
                shuffled_idx = self.rng.permutation(len(sentences))
                sentences = [sentences[i] for i in shuffled_idx]
                labels = [labels[i] for i in shuffled_idx]

            self.generated_document_list = [sentences]
            self.generated_document_labels_list = [labels]
            self.generated_document_group_list = [group_label]
