"""
Utilities for creating the dataset for 

"""
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import collections
import pandas as pd
import numpy as np
import string
from tqdm import tqdm, notebook


class Datasplitter(object):
    """
    Helper class for generating the datasplits and batches
    """

    @staticmethod
    def generate_batches(
        dataset, batch_size, shuffle=True, drop_last=False, device="cpu"
    ):
        """
        A generator function which wraps the PyTorch DataLoader. It will 
        ensure each tensor is on the write device location.
        """
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

    @staticmethod
    def get_split_df(data_df, proportions, test=True):
        """Splits the data into train,val and test by label
        Args:
            data_df (pd.DataFrame): DataFrame containing the labels and the samples
            proportions (dict): Should contain the proportions of the respective splits
                as train_proportion, val_proportion, test_proportion
                if val_proportion is not given, it is derived from train_proportion
            test (bool): If true, makes the test split as well. test_proportion
                must be given.
            toy (float): If not None, creates the split on a subset of the
                data_df based on the given float value
        Returns:
            pd.DataFrame with the a column denoting to which split it belongs to
        """
        assert isinstance(data_df, pd.DataFrame)
        assert "label" in data_df
        if test:
            assert "val_proportion" in proportions
            assert "test_proportion" in proportions

        assert isinstance(proportions, dict)
        assert "train_proportion" in proportions
        assert proportions["train_proportion"] <= 1.0

        if "val_proportion" not in proportions:
            proportions["val_proportion"] = 1.0 - proportions["train_proportion"]
        assert sum(val for val in proportions.values()) == 1.0

        by_label = collections.defaultdict(list)
        for _, row in notebook.tqdm(data_df.iterrows(), total = len(data_df)):
            by_label[row.label].append(row.to_dict())

        final_list = []
        for _, item_list in tqdm(sorted(by_label.items())):
            np.random.shuffle(item_list)

            n_total = len(item_list)
            n_train = int(proportions["train_proportion"] * n_total)
            n_val = int(proportions["val_proportion"] * n_total)

            # Give data point a split attribute
            for item in item_list[:n_train]:
                item["split"] = "train"

            for item in item_list[n_train : n_train + n_val]:
                item["split"] = "val"
            if test:
                n_test = int(proportions["test_proportion"] * n_total)
                for item in item_list[n_train + n_val :]:
                    item["split"] = "test"

            final_list.extend(item_list)

        final_df = pd.DataFrame(final_list)
        return final_df


class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = dict()
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def add_token(self, token):
        """Update mapping dicts based on the token.

            Args:
                token (str): the item to add into the Vocabulary
            Returns:
                index (int): the integer corresponding to the token
            """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def get_max_len(self):
        return self._max_len

    def set_max_len(self, max_len):
        """Sets the maximum length seen so far in the vocabulary
        Args:
            max_len (int)
        """
        self._max_len = max_len

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

            Args:
                tokens (list): a list of string tokens
            Returns:
                indices (list): a list of indices corresponding to the tokens
            """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token 
            or the UNK index if token isn't present.

            Args:
                token (str): the token to look up 
            Returns:
                index (int): the index corresponding to the token
            Notes:
                `unk_index` needs to be >=0 (having been added into the Vocabulary) 
                for the UNK functionality 
            """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

            Args: 
                index (int): the index to look up
            Returns:
                token (str): the token corresponding to the index
            Raises:
                KeyError: if the index is not in the Vocabulary
            """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(
        self,
        token_to_idx=None,
        add_seq_token=True,
        unk_token="<UNK>",
        mask_token="<MASK>",
        begin_seq_token="<BEGIN>",
        end_seq_token="<END>",
    ):

        super(SequenceVocabulary, self).__init__(token_to_idx, add_unk=False)

        self.add_seq_token = add_seq_token

        self._mask_token = mask_token
        self._unk_token = unk_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)

        if self.add_seq_token:
            self._begin_seq_token = begin_seq_token
            self._end_seq_token = end_seq_token
            self.begin_seq_index = self.add_token(self._begin_seq_token)
            self.end_seq_index = self.add_token(self._end_seq_token)

    def set_max_len(self, max_len):
        """Sets the maximum length seen so far in the vocabulary
        Args:
            max_len (int)
        """
        super(SequenceVocabulary, self).set_max_len(max_len)
        if self.add_seq_token:
            self._max_len = self._max_len + 2

    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class SequenceVectorizer(object):
    """Coordinates the Vocabularies and puts them to use
    WARNING: use from_dataframe for stable performance
    """
    def __init__(self, data_vocab, label_vocab, vector_len=-1, mode="onehot"):
        """
        Args:
            data_vocab (SequenceVocabulary): maps words to integers
            label_vocab (Vocabulary): maps class labels to integers
            vector_len (int): length of the vector produced by the vectorizer.
                if -1, the size of the vector is the size of the sentence being 
                vectorized
            mode (str): one of "onehot","embd". If "embd", the tokens formed 
                after vectorization are replaced with their index 
                in the Vocabulary._token_to_idx.keys()
        """
        assert mode in ["onehot", "embd"]
        assert vector_len != 0

        self.data_vocab = data_vocab
        self.label_vocab = label_vocab
        self.vector_len = vector_len
        self.mode = mode

    def get_words(self):
        return self.data_vocab._token_to_idx.keys()

    def vectorize(self, sent):
        """Creates a vector from a given sentence
        Args:
            sent (str): a sentence
            tokenizer (callable): function to split :sent: into tokens

        >>> data_vocab =  Vocabulary()
        >>> sent = "hello bro i am a going home"
        >>> idx = data_vocab.add_many(sent.strip().split())
        >>> v = Vectorizer(data_vocab,None)
        >>> print(v.vectorize(sent))
        [0. 1. 1. 1. 1. 1. 1. 1.]
        """
        assert isinstance(sent, str)

        sent = sent.split()
        if self.mode == "onehot":
            out_vector = np.zeros(len(self.data_vocab), dtype=np.float32)
            for token in sent:
                out_vector[self.data_vocab.lookup_token(token)] = 1
            return out_vector

        elif self.mode == "embd":
            indices = [self.data_vocab.lookup_token(token) for token in sent]
            # BUG what if adding seq tokens
            # causes indices to be more than vector_len ?
            indices = [self.data_vocab.begin_seq_index] + indices
            indices.append(self.data_vocab.end_seq_index)

            if self.vector_len < 0:
                self.vector_len = len(indices)

            out_vector = np.zeros(self.vector_len, dtype=np.int64)
            out_vector[: len(indices)] = indices
            out_vector[len(indices):] = self.data_vocab.mask_index
        return out_vector

    @classmethod
    def from_dataframe(
        cls,
        data_df,
        cutoff=25,
        tokenizer=None,
        vector_len=None,
        mode="onehot",
        **vocab_args,
    ):
        """Instantiates the vector from the dataset dataframe

        Args:
            data_df (pd.DataFrame): the dataset dataframe
            cutoff (int): frequencey based filtering
            tokenizer (Callable): generates tokens from sentences 
            vector_len (int,str): length of the vector produced by the vectorizer.
                -1 : the size of the vector is the size of the sentence being vectorized
                max : the size is set to the no of tokens in the longest sentence
                    in the dataframe

        Returns:
            an instance of the ReviewVectorizer
        """
        label_vocab = Vocabulary(add_unk=False)

        data_vocab = SequenceVocabulary(**vocab_args)

        if tokenizer == None:
            def tokenizer(x):
                return x.split()

        # Add labels
        for label in sorted(set(data_df.label)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        max_len = -1e10
        word_counts = collections.Counter()
        for sent in data_df.text:
            tokenized_sent = tokenizer(sent)
            if len(tokenized_sent) > max_len:
                max_len = len(tokenized_sent)
            for token in tokenized_sent:
                if token not in string.punctuation:
                    word_counts[token] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                data_vocab.add_token(word)
        if vector_len == None:
            vector_len = -1
        elif vector_len == "max":
            data_vocab.set_max_len(max_len)
            vector_len = data_vocab.get_max_len()

        return cls(data_vocab, label_vocab, vector_len, mode)

    @classmethod
    def from_dataframe_nofilter(
        cls,
        data_df,
        tokenizer=None,
        vector_len=None,
        mode="onehot",
        **vocab_args,
    ):
        """Instantiates the vector from the dataset dataframe
        Add all tokens in a sentence without any filters

        Args:
            data_df (pd.DataFrame): the dataset dataframe
            cutoff (int): frequencey based filtering
            tokenizer (Callable): generates tokens from sentences 
            vector_len (int,str): length of the vector produced by the vectorizer.
                -1 : the size of the vector is the size of the sentence being vectorized
                max : the size is set to the no of tokens in the longest sentence
                    in the dataframe

        Returns:
            an instance of the ReviewVectorizer
        """
        label_vocab = Vocabulary(add_unk=False)

        data_vocab = SequenceVocabulary(**vocab_args)

        if tokenizer == None:
            def tokenizer(x):
                return x.split()

        # Add labels
        for label in sorted(set(data_df.label)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        max_len = -1e10
        for sent in data_df.text:
            sent = str(sent)
            tokenized_sent = tokenizer(sent)
            if len(tokenized_sent) > max_len:
                max_len = len(tokenized_sent)
            data_vocab.add_many(tokenized_sent)

        if vector_len == None:
            vector_len = -1
        elif vector_len == "max":
            data_vocab.set_max_len(max_len)
            vector_len = data_vocab.get_max_len()

        return cls(data_vocab, label_vocab, vector_len, mode)

class Vectorizer(object):
    """Coordinates the Vocabularies and puts them to use
    WARNING: use from_dataframe for stable performance
    """

    def __init__(self, data_vocab, label_vocab, vector_len=-1, mode="onehot"):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
            rating_vocab (Vocabulary): maps class labels to integers
            vector_len (int): length of the vector produced by the vectorizer.
                if -1, the size of the vector is the size of the sentence being 
                vectorized
            mode (str): one of "onehot","embd". If "embd", the tokens formed 
                after vectorization are replaced with their index 
                in the Vocabulary._token_to_idx.keys()
        """
        assert mode in ["onehot", "embd"]
        assert vector_len != 0

        self.data_vocab = data_vocab
        self.label_vocab = label_vocab
        self.vector_len = vector_len
        self.mode = mode

    def get_words(self):
        return self.data_vocab._token_to_idx.keys()

    def vectorize(self, sent):
        """Creates a vector from a given sentence
        Args:
            sent (str): a sentence
            tokenizer (callable): function to split :sent: into tokens

        >>> data_vocab =  Vocabulary()
        >>> sent = "hello bro i am a going home"
        >>> idx = data_vocab.add_many(sent.strip().split())
        >>> v = Vectorizer(data_vocab,None)
        >>> print(v.vectorize(sent))
        [0. 1. 1. 1. 1. 1. 1. 1.]
        """
        assert isinstance(sent, str)

        sent = sent.split()
        if self.mode == "onehot":
            out_vector = np.zeros(len(self.data_vocab), dtype=np.float32)
            for token in sent:
                if token not in string.punctuation:
                    out_vector[self.data_vocab.lookup_token(token)] = 1
            return out_vector

        elif self.mode == "embd":
            indices = [self.data_vocab.lookup_token(token) for token in sent]
            # BUG what if adding seq tokens
            # causes indices to be more than vector_len ?
            if (
                isinstance(self.data_vocab, SequenceVocabulary)
                and self.data_vocab.add_seq_token
            ):
                print('hello')
                indices = [self.data_vocab.begin_seq_index] + indices
                indices.append(self.data_vocab.end_seq_index)

            if self.vector_len < 0:
                self.vector_len = len(indices)

            out_vector = np.zeros(self.vector_len, dtype=np.int64)
            out_vector[: len(indices)] = indices
            out_vector[len(indices) :] = self.data_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(
        cls,
        data_df,
        cutoff=25,
        tokenizer=None,
        vector_len=None,
        mode="onehot",
        vocab_class=Vocabulary,
        **vocab_args,
    ):
        """Instantiates the vector from the dataset dataframe

        Args:
            data_df (pd.DataFrame): the dataset dataframe
            cutoff (int): frequencey based filtering
            tokenizer (Callable): generates tokens from sentences 
            vector_len (int,str): length of the vector produced by the vectorizer.
                -1 : the size of the vector is the size of the sentence being vectorized
                max : the size is set to the no of tokens in the longest sentence
                    in the dataframe

        Returns:
            an instance of the ReviewVectorizer
        """
        label_vocab = Vocabulary(add_unk=False)

        data_vocab = vocab_class(**vocab_args)

        if tokenizer == None:
            def tokenizer(x):
                return x.split()

        # Add labels
        for label in sorted(set(data_df.label)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        max_len = -1e10
        word_counts = collections.Counter()
        for sent in data_df.text:
            tokenized_sent = tokenizer(sent)
            if len(tokenized_sent) > max_len:
                max_len = len(tokenized_sent)
            for token in tokenized_sent:
                if token not in string.punctuation:
                    word_counts[token] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                data_vocab.add_token(word)
        if vector_len == None:
            vector_len = -1
        elif vector_len == "max":
            data_vocab.set_max_len(max_len)
            vector_len = data_vocab.get_max_len()

        return cls(data_vocab, label_vocab, vector_len, mode)
    
    @classmethod
    def from_dataframe_nofilter(
        cls,
        data_df,
        tokenizer=None,
        vector_len=None,
        mode="onehot",
        vocab_class=Vocabulary,
        **vocab_args,
    ):
        """Instantiates the vector from the dataset dataframe
        Add all tokens in a sentence without any filters

        Args:
            data_df (pd.DataFrame): the dataset dataframe
            cutoff (int): frequencey based filtering
            tokenizer (Callable): generates tokens from sentences 
            vector_len (int,str): length of the vector produced by the vectorizer.
                -1 : the size of the vector is the size of the sentence being vectorized
                max : the size is set to the no of tokens in the longest sentence
                    in the dataframe

        Returns:
            an instance of the ReviewVectorizer
        """
        label_vocab = Vocabulary(add_unk=False)

        data_vocab = vocab_class(**vocab_args)

        if tokenizer == None:
            def tokenizer(x):
                return x.split()

        # Add labels
        for label in sorted(set(data_df.label)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        max_len = -1e10
        for sent in data_df.text:
            tokenized_sent = tokenizer(sent)
            if len(tokenized_sent) > max_len:
                max_len = len(tokenized_sent)
            data_vocab.add_many(tokenized_sent)

        if vector_len == None:
            vector_len = -1
        elif vector_len == "max":
            data_vocab.set_max_len(max_len)
            vector_len = data_vocab.get_max_len()

        return cls(data_vocab, label_vocab, vector_len, mode)


class Datasetmaker(Dataset):
    def __init__(self, data_df, vectorizer):
        """
        Args:
            data_df (pd.DataFrame) : the dataset
            vectorizer (Vectorizer) : vectorizer instantiated from the dataset
        """
        self.data_df = data_df
        self._vectorizer = vectorizer

        self.train_df = self.data_df[self.data_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == "val"]
        self.val_size = len(self.val_df)

        self.test_df = self.data_df[self.data_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

    def get_vectorizer(self):
        """returns the vectorizer"""
        return self._vectorizer

    def set_split(self, split="train"):
        """Selects the splits in the dataset using the column in the dataframe

        Args:
            split (str): one of 'train','test' or 'val
        """
        assert split in {"train", "test", "val"}
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) 
            and label (y_target)
        """
        row = self._target_df.iloc[index]

        data_vector = self._vectorizer.vectorize(row.text)

        label_index = self._vectorizer.label_vocab.lookup_token(row.label)

        x_index = index

        return {"x_data": data_vector, "y_target": label_index, "x_index": index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size(int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


if __name__ == "__main__":
    data_vocab = Vocabulary()
    sent = "hello bro i am a going home"
    idx = data_vocab.add_many(sent.strip().split())
    v = Vectorizer(data_vocab, None)
    print(v.vectorize(sent, mode="onehot"))
    sent2 = "nlg hello bro i am a going home"
    print(v.vectorize(sent2, mode="onehot"))
