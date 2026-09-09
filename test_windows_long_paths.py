"""Windows long-path regression tests for generated pipeline artifacts."""

import os
import tempfile
import unittest
import uuid
from pathlib import Path

import pandas as pd

from base_mvpipeline import _path_for_io


@unittest.skipUnless(os.name == 'nt', 'Windows-specific path behavior')
class WindowsLongPathTests(unittest.TestCase):
    def test_dataframe_can_be_written_beyond_legacy_max_path(self):
        test_root = Path(tempfile.gettempdir()) / f'grower-long-path-{uuid.uuid4().hex}'
        first_level = test_root / ('a' * 100)
        long_directory = first_level / ('b' * 100)
        output_path = long_directory / 'ground_truth_quality_report.csv'

        try:
            os.makedirs(_path_for_io(long_directory), exist_ok=False)
            self.assertGreater(len(str(output_path)), 260)
            pd.DataFrame({'count': [1]}).to_csv(
                _path_for_io(output_path), index=False
            )
            self.assertTrue(os.path.isfile(_path_for_io(output_path)))
        finally:
            if os.path.isfile(_path_for_io(output_path)):
                os.remove(_path_for_io(output_path))
            for directory in (long_directory, first_level, test_root):
                if os.path.isdir(_path_for_io(directory)):
                    os.rmdir(_path_for_io(directory))


if __name__ == '__main__':
    unittest.main()
