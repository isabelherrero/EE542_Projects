"""Quick checks for the required numerical steps."""
import unittest
import numpy as np
from custom_embedding import train, histogram


class Part5Tests(unittest.TestCase):
    def test_dimensions_and_normalization(self):
        chunks = [{"text": "tcp network packet"}, {"text": "udp network packet"}]
        model, centers, clusters, idf, settings = train(chunks, n=10, k=2)
        self.assertEqual(model.wv.vectors.shape[1], 10)
        self.assertEqual(centers.shape, (2, 10))
        vec = histogram("tcp network packet", clusters, idf, settings)
        self.assertEqual(len(vec), 2)
        self.assertAlmostEqual(float(vec.sum()), 1.0)
        self.assertTrue(np.all(vec >= 0))


if __name__ == "__main__":
    unittest.main()
