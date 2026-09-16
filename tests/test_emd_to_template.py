import os
import sys
import gzip
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from polnet import lio
from polnet.stomo import MmerFile
from polnet.utils.utils import bin_volume, lowpass_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "data_gen", "templates"))
import emd_to_template  # noqa


class TestBinVolume(unittest.TestCase):

    def test_downsample_updates_shape_and_voxel_size(self):
        volume = np.random.rand(20, 20, 20).astype(np.float32)
        out, v_size = bin_volume(volume, voxel_size=2.0, bin_factor=2.0)
        self.assertEqual(out.shape, (10, 10, 10))
        self.assertEqual(v_size, 4.0)

    def test_label_volume_uses_nearest_neighbor(self):
        volume = np.zeros((10, 10, 10), dtype=np.float32)
        volume[5:, :, :] = 1.0
        out, v_size = bin_volume(volume, voxel_size=1.0, bin_factor=2.0, is_label=True)
        self.assertEqual(out.shape, (5, 5, 5))
        self.assertEqual(v_size, 2.0)
        self.assertTrue(set(np.unique(out)).issubset({0.0, 1.0}))


class TestLowpassFilter(unittest.TestCase):

    def test_smooths_a_sharp_edge(self):
        tomo = np.zeros((20, 20, 20), dtype=np.float32)
        tomo[10:] = 1.0
        out = lowpass_filter(tomo, source_vsize=10.0, target_vsize=10.0)
        self.assertLess(out[10, 10, 10], 1.0)
        self.assertGreater(out[9, 10, 10], 0.0)

    def test_higher_target_vsize_blurs_more(self):
        tomo = np.zeros((20, 20, 20), dtype=np.float32)
        tomo[10:] = 1.0
        mild = lowpass_filter(tomo, source_vsize=10.0, target_vsize=10.0)
        strong = lowpass_filter(tomo, source_vsize=10.0, target_vsize=30.0)
        self.assertLess(strong.std(), mild.std())


class TestPadMargin(unittest.TestCase):

    def test_pads_by_given_voxels_on_every_side(self):
        tomo = np.ones((5, 5, 5), dtype=np.float32)
        out = emd_to_template.pad_margin(tomo, pad_voxels=10)
        self.assertEqual(out.shape, (25, 25, 25))
        self.assertEqual(out[0, 0, 0], 0.0)
        self.assertTrue(np.all(out[10:15, 10:15, 10:15] == 1.0))


class TestNormalize(unittest.TestCase):

    def test_negative_background_clips_to_zero(self):
        tomo = np.zeros((100, 10, 10), dtype=np.float32)
        tomo[:50] = -5.0
        tomo[50:] = np.linspace(0, 1, 50)[:, None, None]
        out = emd_to_template.normalize(tomo)
        self.assertTrue(np.all(out[:50] == 0.0))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_clips_positive_outlier_before_scaling(self):
        tomo = np.full((10, 10, 10), 0.2, dtype=np.float32)
        tomo[0, 0, 0] = 1000.0
        out = emd_to_template.normalize(tomo)
        self.assertLessEqual(out.max(), 1.0)
        self.assertGreater(out[5, 5, 5], 0.0)


class TestComputeIso(unittest.TestCase):

    def test_override_takes_precedence(self):
        tomo = np.random.rand(10, 10, 10).astype(np.float32)
        self.assertEqual(emd_to_template.compute_iso(tomo, override=0.42), 0.42)

    def test_otsu_separates_two_populations(self):
        tomo = np.zeros((20, 10, 10), dtype=np.float32)
        tomo[10:] = 1.0
        iso = emd_to_template.compute_iso(tomo, override=None)
        self.assertGreater(iso, 0.0)
        self.assertLess(iso, 1.0)


class TestZeroSolvent(unittest.TestCase):

    def test_keeps_largest_component_zeros_isolated_noise(self):
        tomo = np.zeros((30, 10, 10), dtype=np.float32)
        tomo[2:8] = 0.8  # main structure
        tomo[20:23] = 0.9  # isolated far-field noise blob, well outside the margin
        out = emd_to_template.zero_solvent(tomo, iso=0.5)
        self.assertTrue(np.all(out[2:8] == 0.8))
        self.assertTrue(np.all(out[20:23] == 0.0))

    def test_keeps_margin_around_structure(self):
        tomo = np.zeros((30, 10, 10), dtype=np.float32)
        tomo[10:15] = 0.8
        tomo[9] = 0.3  # sub-threshold penumbra right next to the structure
        out = emd_to_template.zero_solvent(tomo, iso=0.5)
        self.assertAlmostEqual(out[9, 5, 5], 0.3, places=5)


class TestWritePns(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_round_trips_through_mmerfile(self):
        args = mock.Mock(pmer_l=1.3, pmer_l_max=2.0, pmer_occ=0.045, pmer_over_tol=0.002)
        pns_path = os.path.join(self.tmp_dir, "emd_1234_10A.pns")
        emd_to_template.write_pns(
            pns_path, "emd_1234", "/templates/mrcs_10A/emd_1234.mrc", 0.12, "ribosome", args
        )

        loaded = MmerFile(pns_path)
        self.assertEqual(loaded.get_mmer_id(), "emd_1234")
        self.assertEqual(loaded.get_mmer_svol(), "/templates/mrcs_10A/emd_1234.mrc")
        self.assertEqual(loaded.get_iso(), 0.12)
        self.assertEqual(loaded.get_pmer_l(), 1.3)
        self.assertEqual(loaded.get_pmer_l_max(), 2.0)
        self.assertEqual(loaded.get_pmer_occ(), 0.045)
        self.assertEqual(loaded.get_pmer_over_tol(), 0.002)


class TestGenerateTemplate(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.map_path = os.path.join(self.tmp_dir, "emd_1234.map.gz")
        tomo = np.random.rand(8, 8, 8).astype(np.float32)
        fd, raw_path = tempfile.mkstemp(suffix=".mrc")
        os.close(fd)
        lio.write_mrc(tomo, raw_path, v_size=5.0)
        with open(raw_path, "rb") as f_in, gzip.open(self.map_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(raw_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_writes_mrc_and_pns_without_network(self):
        args = mock.Mock(
            emd_id="1234", label=None, out_root=self.tmp_dir, voxel_size=10.0, lowpass=30.0, pad_voxels=10,
            mmer_iso=None, pmer_l=1.2, pmer_l_max=1.0, pmer_occ=0.05, pmer_over_tol=0.001,
        )
        with mock.patch.object(emd_to_template, "download_map", return_value=self.map_path):
            emd_to_template.generate_template(args)

        mrc_path = os.path.join(self.tmp_dir, "templates", "mrcs_10A", "emd_1234.mrc")
        pns_path = os.path.join(self.tmp_dir, "in_10A", "emd_1234_10A.pns")
        self.assertTrue(os.path.isfile(mrc_path))
        self.assertTrue(os.path.isfile(pns_path))

        protein = MmerFile(pns_path)
        self.assertEqual(protein.get_mmer_id(), "emd_1234")
        tomo = lio.load_mrc(mrc_path)
        self.assertEqual(tomo.shape, (24, 24, 24))


if __name__ == "__main__":
    unittest.main()
