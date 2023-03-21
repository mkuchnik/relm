"""Tests for indexing relm."""

import unittest

import numpy as np

import relm.indexing


class TestIndexing(unittest.TestCase):
    """Test that indexing utilities work."""

    def test_sort(self):
        """Test that sorting utilities find top elements."""
        a = list(range(100))
        expected_top_10 = set(a[-10:])
        found_top_10 = set(relm.indexing.top_N_elements_idx(a, 10))
        self.assertEqual(expected_top_10, found_top_10)
        for i, x in enumerate(a):
            found_i = relm.indexing.get_element_rank(a, x)
            self.assertEqual(found_i, i + 1)  # 0-indexing difference

    def test_rank_array(self):
        """Test that ranking elements of an array works."""
        a = np.arange(10)
        expected_a_ranks = a + 1
        a_ranks = relm.indexing.rank_array(a)
        self.assertTrue(isinstance(a_ranks, np.ndarray))
        self.assertEqual(expected_a_ranks.shape, a_ranks.shape)
        self.assertEqual(expected_a_ranks.tolist(), a_ranks.tolist())

    def test_get_top_ranks(self):
        """Test that the top k ranks are returned."""
        n = 10
        probs = np.zeros((n, n))
        probs += np.eye(n)
        prob_ranks = relm.indexing.get_top_k_ranks(probs)
        for i in range(prob_ranks.shape[0]):
            ranks = prob_ranks[i, :]
            expected_ranks = list(range(2, n + 1))
            expected_ranks.insert(i, 1)
            expected_ranks = np.array(expected_ranks)
            self.assertEqual(expected_ranks.tolist(), ranks.tolist())

    def test_get_top_k(self):
        """Test that the top k indicies are returned."""
        n = 10
        probs = np.zeros((n, n))
        probs += np.eye(n)
        ind = relm.indexing.get_top_k_mask(probs, k=1)
        expected_ind = np.arange(n)[:, None]
        self.assertEqual(expected_ind.tolist(), ind.tolist())
        probs = np.zeros((n, n))
        probs[:, 0] = 2
        probs[:, 1] = 1
        ind = relm.indexing.get_top_k_mask(probs, k=2)
        expected_ind = np.array([0, 1])[None, :]
        expected_ind = np.repeat(expected_ind, n, axis=0)
        self.assertEqual(expected_ind.tolist(), ind.tolist())
        probs = np.arange(n)[None, :]
        ind = relm.indexing.get_top_k_mask(probs, k=n)
        expected_ind = np.arange(n)[None, ::-1]
        self.assertEqual(expected_ind.tolist(), ind.tolist())
