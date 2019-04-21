import sys
import unittest
from mPTB.class_dataset import ClassDataset
from mPTB.tokenization_sentencepiece import FullTokenizer


class ClassDatasetTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FullTokenizer(
            "sample_text.model", "sample_text.vocab")

    def test_instance(self):
        dataset = ClassDataset(
            self.tokenizer,
            max_pos=128,
            label_num=3,
            dataset_path="sample_text_class.txt",
            header_skip=False
        )
        records_num = 0
        with open("sample_text.txt",  encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    records_num += 1
        self.assertEqual(records_num, len(dataset))

    def test_not_found_file_load(self):
        with self.assertRaises(FileNotFoundError):
            ClassDataset(
                self.tokenizer,
                max_pos=128,
                label_num=3,
                dataset_path="sample_text_clazz.txt",
                header_skip=False
            )

    def test_delimiter_two_pattern_file_load(self):
        with self.assertRaises(AssertionError):
            with self.assertRaises(FileNotFoundError):
                ClassDataset(
                    self.tokenizer,
                    max_pos=128,
                    label_num=3,
                    dataset_path="sample_text_class_delim_error.txt",
                    header_skip=False
                )

    def test_label_not_match_file_load(self):
        with self.assertRaises(AssertionError):
            ClassDataset(
                self.tokenizer,
                max_pos=128,
                label_num=2,
                dataset_path="sample_text_class.txt",
                header_skip=False
            )


if __name__ == '__main__':
    unittest.main()
