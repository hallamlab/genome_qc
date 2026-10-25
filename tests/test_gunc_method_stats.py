import unittest
import numpy as np
import pandas as pd
from gunc_method_stats import gunc_tests

class GuncTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for i in range(40):
            for category, positive in [('SAGs',i%10==0),('xPG_SAGs',i%4==0)]:
                state='credible_chimeric_signal' if positive else 'likely_not_chimeric'
                if i%7==1: state='low_reference_uncertain'
                if i%11==2: state='unscored'
                rows.append(dict(sample='s'+str(i%2),category=category,Genome_Id=f'AB-{i}_A',
                                 gunc_strict_assessment=state))
        frame=pd.DataFrame(rows)
        frame.index=np.arange(len(frame))*3
        return frame

    def test_denominators_and_linkage(self):
        data,tests,counts=gunc_tests(self.frame())
        unknown=data.gunc_strict_assessment.isin(['unscored','low_reference_uncertain'])
        self.assertTrue(data.loc[unknown,'gunc_assessed_signal'].isna().all())
        self.assertTrue(data.loc[unknown,'gunc_detected_signal'].eq(0).all())
        self.assertEqual(data.dependency_group.nunique(),40)
        self.assertEqual(counts.n_all.sum(),80)
        self.assertEqual(counts.n_assessed.sum(),int((~unknown).sum()))
        self.assertEqual(len(tests),4)
        self.assertTrue(tests.status.eq('ok').all())
        self.assertEqual(len(tests.analysis_scope.unique()),4)

    def test_constant_and_invalid(self):
        frame=self.frame()
        frame['gunc_strict_assessment']='likely_not_chimeric'
        _,tests,_=gunc_tests(frame)
        self.assertTrue(tests.pvalue.isna().all())
        self.assertTrue(tests.status.eq('constant_outcome').all())
        frame.iloc[0,frame.columns.get_loc('gunc_strict_assessment')]=None
        with self.assertRaisesRegex(ValueError,'Missing or unknown'):
            gunc_tests(frame)
