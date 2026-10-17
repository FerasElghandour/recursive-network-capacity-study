import unittest
import numpy as np
from followup import holm, paired_summary, analyze


class StatisticsChecks(unittest.TestCase):
    def test_exact_sign_flip_known_oracle(self):
        self.assertEqual(paired_summary([1.,1.,1.,1.])['p_value'],.125)
        self.assertEqual(paired_summary([0.,0.])['p_value'],1.)
        self.assertAlmostEqual(paired_summary([-1.,2.,3.])['mean_difference'],4/3)

    def test_holm_hand_calculation(self):
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])

    def test_incomplete_or_duplicate_pairs_rejected(self):
        p={'width':16,'datasets':['optdigits'],'seeds':[0,1],'orderings':['native','permuted'],'models':['balanced','random']}
        with self.assertRaisesRegex(ValueError,'Incomplete'):analyze([],p)
        r=dict(dataset='optdigits',ordering='native',width=16,seed=0,model='balanced')
        with self.assertRaisesRegex(ValueError,'Duplicate'):analyze([r,r],p)


if __name__=='__main__':unittest.main()
