# test_llm.py

import torch
import unittest
from llm import BigramModel


class TestBigramModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.model = BigramModel(self.vocab_size)

    def test_forward(self):
        idx = torch.tensor([[1, 2, 3]])
        logits, loss = self.model.forward(idx)
        self.assertEqual(logits.shape, (1, 3, self.vocab_size))
        self.assertIsNone(loss)

        targets = torch.tensor([[2, 3, 4]])
        logits, loss = self.model.forward(idx, targets)
        self.assertEqual(logits.shape, (1, 3, self.vocab_size))
        self.assertIsNotNone(loss)

    def test_generate(self):
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated_idx = self.model.generate(idx, max_new_tokens)
        self.assertEqual(generated_idx.shape, (1, 8))

    def test_get_batch(self):
        split = "train"
        x, y = self.model.get_batch(split)
        self.assertEqual(x.shape, (self.model.batch_size, self.model.block_size))
        self.assertEqual(y.shape, (self.model.batch_size, self.model.block_size))

    def test_estimate_loss(self):
        loss_dict = self.model.estimate_loss()
        self.assertIn("train", loss_dict)
        self.assertIn("val", loss_dict)
        self.assertIsInstance(loss_dict["train"], torch.Tensor)
        self.assertIsInstance(loss_dict["val"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
