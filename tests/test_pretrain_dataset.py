import unittest
from mptb.pretrain_dataset import NextSentencePredictionDataset, StackedSentenceDataset
from mptb.tokenization_sentencepiece import FullTokenizer


class PretrainDatasetTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FullTokenizer(
            "sample_text.model", "sample_text.vocab")

    def test_instance(self):
        dataset = NextSentencePredictionDataset(
            self.tokenizer,
            max_pos=128,
            dataset_path="sample_text.txt",
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
            NextSentencePredictionDataset(
                self.tokenizer,
                max_pos=128,
                dataset_path="empty.txt",
                on_memory=True
            )

    def test_not_found_file_load(self):
        with self.assertRaises(FileNotFoundError):
            NextSentencePredictionDataset(
                self.tokenizer,
                max_pos=128,
                dataset_path="not_found_text.txt",
                on_memory=True
            )

    def test_last_row_empty_file_load(self):
        sample_text = NextSentencePredictionDataset(
            self.tokenizer,
            max_pos=128,
            dataset_path="sample_text.txt",
            on_memory=True
        )
        last_row_empty = NextSentencePredictionDataset(
            self.tokenizer,
            max_pos=128,
            dataset_path="last_row_empty.txt",
            on_memory=True
        )
        self.assertEqual(len(sample_text), len(last_row_empty))

    def test_get_item_one(self):
        dataset = NextSentencePredictionDataset(
            self.tokenizer,
            max_pos=128,
            dataset_path="sample_text.txt",
            on_memory=True
        )
        self.assertIsNotNone(dataset.__getitem__(1))

    def test_get_item_use_not_item_index(self):
        with self.assertRaises(AssertionError):
            dataset = NextSentencePredictionDataset(
                self.tokenizer,
                max_pos=128,
                dataset_path="sample_text.txt",
                on_memory=True
            )
            dataset.__getitem__([1, 2])

    def test_stack_instance(self):
        dataset = StackedSentenceDataset(
            self.tokenizer,
            max_pos=128,
            dataset_path="sample_text.txt",
        )
        delim_return = 0
        num_docs = 3
        with open("sample_text.txt",  encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    delim_return += 1
        self.assertLessEqual(len(dataset), delim_return - num_docs)


if __name__ == '__main__':
    unittest.main()
