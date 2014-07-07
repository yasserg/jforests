package edu.uci.jforests.eval.ranking;

import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.util.CDF_Normal;

/** Implements the TRisk risk-sensitive optimisation metric called FARO, as defined by Dincer et al.
 * See Taner Dincer, Craig Macdonald and Iadh Ounis, Hypothesis Testing for Risk-Sensitive Evaluation 
 * of Retrieval Systems, SIGIR 2014.
 * <p><b>Implementation Details</b><p>
 * It is assumed that the natural ordering of documents for each query defines
 * the baseline.
 * @author Craig Macdonald, University of Glasgow; Taner Dincer, Mugla University.
 */
public class TRiskAwareFAROEval extends TRiskAwareSAROEval {

	public TRiskAwareFAROEval(EvaluationMetric _parent, double alpha) {
		super(_parent, alpha);
	}

	class FAROSwapScorer extends SAROSwapScorer
	{
		public FAROSwapScorer(double[] targets, int[] boundaries, int trunc,
				int[][] labelCounts, double _alpha, SwapScorer _parent) {
			super(targets, boundaries, trunc, labelCounts, _alpha, _parent);
		}
		
		@Override
		public double getDelta(int queryIndex, int betterIdx, int rank_i, int worseIdx, int rank_j) 
		{
			//get the change in NDCG
			final double delta_M = parentSwap.getDelta(queryIndex, betterIdx, rank_i, worseIdx, rank_j);

			final double M_m = modelEval[queryIndex];
			final double M_b = baselineEval[queryIndex];

			// Score difference
			double d_i = M_m - M_b;

			final double TRisk = d_i / currPairedSTD;
			
			// beta asymptotically ranges in between a value as small as 0 and a value as large as alpha,
			// proportionally to the level of risk commited by the current topic.
			// This version follows its own way of weighing a given delta.
			// NB: in Dincer et al, beta is called \alpha'
			final double beta = (1 - CDF_Normal.normp(TRisk)) * alpha;

			final double delta_T = (1 + beta) * delta_M;
			
			return delta_T;
		}

		
	}

		
	@Override
	public SwapScorer getSwapScorer(double[] targets, int[] boundaries,
			int trunc, int[][] labelCounts) throws Exception 
	{
		final SwapScorer parentMeasure = ((RankingEvaluationMetric) parent).getSwapScorer(targets, boundaries, trunc, labelCounts);
		return new FAROSwapScorer(targets, boundaries, trunc, labelCounts,
				ALPHA, 
				parentMeasure);
	}
}
