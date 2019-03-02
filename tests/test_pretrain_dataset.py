import sys
import unittest
from pretrain_dataset import PretrainDataset
from tokenization_sentencepiece import FullTokenizer


class PretrainDatasetTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FullTokenizer(
            "sample_text.model", "sample_text.vocab", do_lower_case=True)

    def test_instance(self):
        dataset = PretrainDataset(
            "sample_text.txt",
            self.tokenizer,
            max_pos=128,
            corpus_lines=None,
            on_memory=True
        )
        delim_return = 0
        num_docs = 3
        start_zero = 1  # index 0 start
        with open("sample_text.txt",  encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    delim_return += 1
        self.assertEqual(delim_return-num_docs-start_zero, len(dataset))

    def test_empty_file_load(self):
        with self.assertRaises(ValueError):
            PretrainDataset(
                "empty.txt",
                self.tokenizer,
                max_pos=128,
                corpus_lines=None,
                on_memory=True
            )

    def test_last_row_empty_file_load(self):
        dataset1 = PretrainDataset(
            "sample_text.txt",
            self.tokenizer,
            max_pos=128,
            corpus_lines=None,
            on_memory=True
        )
        dataset2 = PretrainDataset(
            "last_row_empty.txt",
            self.tokenizer,
            max_pos=128,
            corpus_lines=None,
            on_memory=True
        )
        self.assertEqual(len(dataset1), len(dataset2))


if __name__ == '__main__':
    unittest.main()
