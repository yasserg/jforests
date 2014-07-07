package edu.uci.jforests.eval.ranking;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.MathUtil;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;

/** Implements the URisk risk-sensitive evaluation metric, as defined by Wang et al.
 * See Lidan Wang, Paul Bennet and Kevyn Collin-Thompson, SIGIR 2012.
 * <p><b>Implementation Details</b><p>
 * It is assumed that the natural ordering of documents for each query defines
 * the baseline.
 * @author Craig Macdonald, University of Glasgow
 */
public class URiskAwareEval extends RankingEvaluationMetric {

	EvaluationMetric parent;
	final double ALPHA;
	
	static class URiskSwapScorer extends SwapScorer
	{
		/** alpha value in the Utility measure */
		double alpha;
		/** swapscorer of the parent metric */
		SwapScorer parentSwap;
		/** M_m: the performance over all queries of the current model
		 * BEFORE any documents are swapped
		 */
		double[] modelEval;
		/** M_b: the mean performance of the baseline ranking over all queries. */
		double[] baselineEval;
			
		
		public URiskSwapScorer(
				double[] targets, 
				int[] boundaries, 
				int trunc,
				int[][] labelCounts,
				double _alpha,
				SwapScorer _parent
				) 
		{
			super(targets, boundaries, trunc, labelCounts);
			this.alpha = _alpha;
			this.parentSwap = _parent;
		}

		@Override
		public void setCurrentIterationEvaluation(int iteration, double[] nDCG) {
			final double meanNDCG = MathUtil.getAvg(nDCG);
			if (iteration == 0)
			{
				System.err.println("Baseline in iteration 0 has peformance " + meanNDCG);
				baselineEval = nDCG;
				modelEval = new double[baselineEval.length];
			}
			else
			{
				System.err.println("Model in iteration "+iteration+" has peformance " + meanNDCG);
				modelEval = nDCG;
			}
			System.err.println("Iteration " + iteration + " NDCG=" + Arrays.toString(nDCG));
		}
		
		@Override
		public double getDelta(int queryIndex, int betterIdx, int rank_i,
				int worseIdx, int rank_j) {
			
			final double M_m = modelEval[queryIndex];
			final double M_b = baselineEval[queryIndex];
			
			//This follows the notation of Section 4.3.2 in Wang et al. to the fullest extent possible.
			//Assertions and if conditions are used to ensure that the conditions described in the proof
			//are arrived at.
			
			//change in effectiveness
			final double delta_M = parentSwap.getDelta(queryIndex, betterIdx, rank_i, worseIdx, rank_j);
			
			final double rel_i = targets[betterIdx];
			final double rel_j = targets[worseIdx];
			
			
			//Our LambaMART implementation only calls getDelta when i is of higher quality better than j.
			assert rel_i >= rel_j;
			//the upshot is that not all scenarios in the proof can ever occur. we now check that conditions
			//are as expected
			
			//hence, delta_M should always be zero or positive, if it is later ranked than j. This would be a "good" swap
			if (rank_i > rank_j)
			{
				assert delta_M >= 0 : "rank_i=" + rank_i + " rank_j="+rank_j + " delta_M="+delta_M;
			}
			//hence, delta_M should always be zero or negative, if it is earlier ranked than j. This would be a "bad" swap
			if (rank_i < rank_j)
			{
				assert delta_M <= 0 : "rank_i=" + rank_i + " rank_j="+rank_j + " delta_M="+delta_M;
			}
			
			//change in tradeoff
			final double delta_T;
			
			//Scenario A
			if (M_m <=  M_b)
			{
				//case A1
				if (rel_i > rel_j && rank_i< rank_j)
				{
					assert delta_M < 0;
					delta_T = (1.0d + alpha) * delta_M;
					assert delta_T < 0 : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M + " => delta_T=" + delta_T ;
				}
				//case A2
				else
				{	
					assert rel_i > rel_j && rank_i > rank_j : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M ;
					assert delta_M >= 0 : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M 
							+ " rel_i="+rel_i+" rel_j="+rel_j+" rank_i="+rank_i+" rank_j="+rank_j;
					if (M_b > M_m + delta_M)
					{
						delta_T = (1.0d + alpha) * delta_M;
					}
					else
					{
						assert M_b <= M_m + delta_M  : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M ;
						delta_T = alpha * (M_b - M_m) + delta_M;
					}					
					assert delta_T > 0 : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M 
						+ " rel_i="+rel_i+" rel_j="+rel_j+" rank_i="+rank_i+" rank_j="+rank_j+"  => delta_T=" + delta_T ;
				}
			}
			else //Scenario B
			{
				assert M_m > M_b : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M;
				if( rel_i > rel_j && rank_i < rank_j )
				{
					assert delta_M <= 0 : "rank_i=" + rank_i + " rank_j="+rank_j + " delta_M="+delta_M;
					//case B1
					if (M_b > M_m - Math.abs(delta_M))
					{
						delta_T = alpha * (M_m - M_b) - (1 + alpha) * Math.abs(delta_M);
					}
					else
					{
						assert M_b <= M_m - Math.abs(delta_M) : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M ;
						delta_T = delta_M;
					}
					assert delta_T < 0 : "M_b="+M_b+" M_m="+M_m + " delta_M=" + delta_M 
						+ " rel_i="+rel_i+" rel_j="+rel_j+" rank_i="+rank_i+" rank_j="+rank_j + " => delta_T=" + delta_T ;
				}
				else 
				{
					assert rel_i > rel_j && rank_i > rank_j;
					delta_T = delta_M;
					assert delta_T > 0;
				}
			}
			
			return delta_T;
		}
		
	}
	
	public URiskAwareEval(EvaluationMetric _parent, double alpha)
	{
		super(_parent.largerIsBetter());
		assert _parent.largerIsBetter();
		parent = _parent;
		ALPHA = alpha;
	}
		
	@Override @SuppressWarnings("unused")
	public double measure(double[] predictions, Sample sample) throws Exception {
		
		final RankingSample rankingSample = (RankingSample)sample;
		assert rankingSample.queryBoundaries.length -1 == rankingSample.numQueries;
		final double[] naturalOrder = computeNaturalOrderScores(predictions.length, rankingSample.queryBoundaries);
				
		double[] baselinePerQuery = ((RankingEvaluationMetric) parent).measureByQuery(naturalOrder, sample);
		double[] perQuery = ((RankingEvaluationMetric) parent).measureByQuery(predictions, sample);
		
		double T1 = 0, T2 = 0;
					
		final int queryLength = perQuery.length;
		double F_reward = 0.0d;
		double F_risk = 0.0d;
		for(int i=0;i<queryLength;i++)
		{
			 F_reward += Math.max(0, perQuery[i] - baselinePerQuery[i]);
			 F_risk += Math.max(0,  baselinePerQuery[i] - perQuery[i]);
			 T1 += baselinePerQuery[i];
			 T2 += perQuery[i];			 
		}
		T1 /= (double)queryLength;
		T2 /= (double)queryLength;
		
		F_reward /= (double) queryLength;
		F_risk /= (double) queryLength;
		return F_reward - (1 + ALPHA) * F_risk;
	}

	@Override
	public SwapScorer getSwapScorer(double[] targets, int[] boundaries,
			int trunc, int[][] labelCounts) throws Exception 
	{
		final SwapScorer parentModel = ((RankingEvaluationMetric) parent).getSwapScorer(targets, boundaries, trunc, labelCounts);
		return new URiskSwapScorer(targets, boundaries, trunc, labelCounts,
				ALPHA, 
				parentModel);
	}

	@Override
	public double[] measureByQuery(double[] predictions, Sample sample)
			throws Exception {
		throw new UnsupportedOperationException("Hmmm, not sure how to calculate this one yet");
	}

	@Override
	public EvaluationMetric getParentMetric() {
		return ((RankingEvaluationMetric) parent).getParentMetric();
	}

	public static class TestURiskSwaps 
	{
		@Test public void testTwoQueries() throws Exception
		{
			BlockingThreadPoolExecutor.init(1);
			RankingEvaluationMetric eval = new URiskAwareEval(new NDCGEval(2,  2), 1);
			SwapScorer s = eval.getSwapScorer(
					new double[]{0,1,0,1}, 
					new int[]{0,2,4}, 
					2, 
					new int[][]{new int[]{1,1,0,0,0}, new int[]{1,1,0,0,0}});
			s.setCurrentIterationEvaluation(0, new double[]{0.1,0.2});
			System.err.println(s.getDelta(0, 0, 0, 1, 1));
			
			s = eval.getSwapScorer(
					new double[]{0,1}, 
					new int[]{0,2}, 
					2, 
					new int[][]{new int[]{1,1,0,0,0}});
			s.setCurrentIterationEvaluation(0, new double[]{0.1});
			System.err.println(s.getDelta(0, 0, 0, 1, 1));
			
			assertTrue(true);
		}
	}

	
}
