import sys
import unittest
from class_csv_dataset import BertCsvDataset
from tokenization_sentencepiece import FullTokenizer


class BertCsvDatasetTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FullTokenizer(
            "sample_text.model", "sample_text.vocab", do_lower_case=True)

    def test_instance(self):
        dataset = BertCsvDataset(
            "sample_text_class.txt",
            self.tokenizer,
            max_pos=128,
            label_num=3,
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
            BertCsvDataset(
                "sample_text_clazz.txt",
                self.tokenizer,
                max_pos=128,
                label_num=3,
                header_skip=False
            )

    def test_delimiter_two_pattern_file_load(self):
        with self.assertRaises(AssertionError):
            BertCsvDataset(
                "sample_text_class_delim_error.txt",
                self.tokenizer,
                max_pos=128,
                label_num=3,
                header_skip=False
            )

    def test_label_not_match_file_load(self):
        with self.assertRaises(AssertionError):
            BertCsvDataset(
                "sample_text_class.txt",
                self.tokenizer,
                max_pos=128,
                label_num=2,
                header_skip=False
            )




if __name__ == '__main__':
    unittest.main()
