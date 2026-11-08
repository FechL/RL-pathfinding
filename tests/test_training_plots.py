"""Evidence handling for training charts: partial logs, missing fields, failed runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from plot_training import read_monitor,read_checkpoints,rolling


class TrainingPlotTests(unittest.TestCase):
    def test_historic_monitor_does_not_invent_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training.monitor.csv'
            path.write_text('#{}\nr,l,t\n100,12,1.2\n-10,5,2.0\n4,')
            rows=read_monitor(path)
            self.assertEqual(len(rows),2)
            self.assertNotIn('is_success',rows[0])
            self.assertEqual(rows[1]['reward'],-10.)

    def test_monitor_records_real_success_and_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training.monitor.csv'
            path.write_text('#{}\nr,l,t,is_success,collision\n100,12,1.2,True,False\n-10,5,2.0,False,True\n')
            rows=read_monitor(path)
            self.assertTrue(rows[0]['is_success'])
            self.assertTrue(rows[1]['collision'])
            self.assertFalse(rows[1]['is_success'])

    def test_checkpoint_exact_duplicates_only(self):
        rows=[dict(stage='checkpoint',steps=20,runs=3,successes=3,mean_time_s=52.),
              dict(stage='final',steps=20,runs=3,successes=3,mean_time_s=52.),
              dict(stage='new',steps=40,runs=3,successes=0,mean_time_s=2.)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoints.json';path.write_text(json.dumps(rows))
            result=read_checkpoints(path)
            self.assertEqual(len(result),2)
            self.assertEqual(result[1]['successes'],0)
            rows[1]['mean_time_s']=53.;path.write_text(json.dumps(rows))
            self.assertEqual(len(read_checkpoints(path)),3)
            rows[0]['successes']=4;path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):read_checkpoints(path)

    def test_rolling_window_uses_available_history(self):
        self.assertEqual(rolling([10,20,30,40],3).tolist(),[10,15,20,30])


if __name__=='__main__':unittest.main()
