package edu.uci.jforests.eval.ranking;

//import static org.junit.Assert.assertEquals;
//import org.junit.Test;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.CDF_Normal;
import edu.uci.jforests.util.MathUtil;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;

/** Implements the TRisk risk-sensitive optimisation metric called SARO, as defined by Dincer et al.
 * See Taner Dincer, Craig Macdonald and Iadh Ounis, Hypothesis Testing for Risk-Sensitive Evaluation 
 * of Retrieval Systems, SIGIR 2014.
 * <p><b>Implementation Details</b><p>
 * It is assumed that the natural ordering of documents for each query defines
 * the baseline.
 * @author Craig Macdonald, University of Glasgow; Taner Dincer, Mugla University.
 */
public class TRiskAwareSAROEval extends URiskAwareEval {

	public TRiskAwareSAROEval(EvaluationMetric _parent, double alpha) {
		super(_parent, alpha);
	}

	class SAROSwapScorer extends URiskSwapScorer
	{
		double currPairedSTD = 0.0d;
		double c;
		double baselineMean;
		
		
		public SAROSwapScorer(double[] targets, int[] boundaries, int trunc,
				int[][] labelCounts, double _alpha, SwapScorer _parent) {
			super(targets, boundaries, trunc, labelCounts, _alpha, _parent);
			c = ((double) boundaries.length -1);
		}
		
		@Override
		/* Basic Version */
		public double getDelta(int queryIndex, int betterIdx, int rank_i, int worseIdx, int rank_j) 
		{
			//get the change in NDCG
			final double delta_M = parentSwap.getDelta(queryIndex, betterIdx, rank_i, worseIdx, rank_j);

			final double M_m = modelEval[queryIndex];
			final double M_b = baselineEval[queryIndex];
			final double rel_i = targets[betterIdx];
			final double rel_j = targets[worseIdx];
	
			// Score difference
			double d_i = M_m - M_b;

			final double TRisk = d_i / currPairedSTD;
			
			// beta asymptotically ranges in between a value as small as 0 and a value as large as alpha,
			// proportionally to the level of risk commited by the current topic.
			// This version follows the way how URisk weighs a given delta.
			//NB: in Dincer et al, beta is called \alpha'
			final double beta = (1 - CDF_Normal.normp(TRisk)) * alpha;

			final double delta_T;
			
			//Scenarios are as defined by Wang et al, SIGIR 2012.
			//Scenario A
			if (M_m <=  M_b)
			{
				//case A1
				if (rel_i > rel_j && rank_i < rank_j)
				{
					delta_T = (1.0d + beta) * delta_M;
				}
				//case A2
				else
				{	
					if (M_b > M_m + delta_M)
					{
						delta_T = (1.0d + beta) * delta_M;
					}
					else
					{
						delta_T = beta * (M_b - M_m) + delta_M;
					}					
				}
			}
			else //Scenario B
			{
				if( rel_i > rel_j && rank_i < rank_j )
				{
					//case B1
					if (M_b > M_m - Math.abs(delta_M))
					{
						delta_T = beta * (M_m - M_b) - (1 + beta) * Math.abs(delta_M);
					}
					else
					{
						delta_T = delta_M;
					}
				}
				else 
				{
					delta_T = delta_M;
				}
			}

			return delta_T;
		}

		@Override
		public void setCurrentIterationEvaluation(int iteration, double[] nDCG) {
			super.setCurrentIterationEvaluation(iteration, nDCG);
			if (iteration == 0)
			{
		        double[] params = getEstimates(baselineEval, modelEval, this.alpha);
				currPairedSTD = Math.sqrt(params[1]);
				System.err.println("Iteration 0 Paired STD="+ currPairedSTD);
				baselineMean = MathUtil.getAvg(baselineEval);
				System.err.println("Iteration 0 NDCG=" + Arrays.toString(nDCG));
			}
			else
			{
				final double modelMean = MathUtil.getAvg(nDCG);
//				if (modelMean < baselineMean)
//				{
//					/* This does not allign with the paper. std is re-evaluated when modelMean < baselineMean. Why? */  
//			        double[] params = getEstimates(baselineEval, modelEval, this.alpha);
//					currPairedSTD = Math.sqrt(params[1]);	
//					System.err.println("Iteration " + iteration + " Paired STD=" + currPairedSTD);
//				}
//				else
//				{
//					System.err.println("Iteration " + iteration + " Paired STD unchanged " + currPairedSTD);
//				}
				System.err.println("Iteration " + iteration + " NDCG=" + Arrays.toString(nDCG));
			}
		}	
	}

	/** returns double[] params where params[0] = URisk; params[1] = PairedVar. */
	public static double[] getEstimates(final double[] baselinePerQuery, final double[] perQuery, final double ALPHA)
	{
        final double c = baselinePerQuery.length;
        double sum = 0D;
        double SSQR = 0D;
        double d_i = 0D;

        for(int i=0; i < c; i++)
        {
            if (perQuery[i] > baselinePerQuery[i])
                d_i = perQuery[i] - baselinePerQuery[i];
            else
                d_i = (1 + ALPHA) * (perQuery[i] - baselinePerQuery[i]);
            sum += d_i;
            SSQR += d_i * d_i;
        }

        final double URisk = sum /c;
        final double SQRS = sum * sum; 
        final double pairedVar = SSQR == SQRS ? 0 : ( SSQR - (SQRS / c) ) / (c-1);
        return new double[] {URisk, pairedVar};
	}
	
	/* Returns TRisk =  Math.sqrt(c / PairedVar) * URisk */
	public static double T_measure(final double[] baselinePerQuery, final double[] perQuery, final double ALPHA)
	{
        final double c = baselinePerQuery.length;

        double[] params = getEstimates(baselinePerQuery, perQuery, ALPHA);
		
		return params[1] == 0D ? 0 : Math.sqrt(c / params[1]) * params[0];   
	}
	
	@Override
	public double measure(double[] predictions, Sample sample) throws Exception {
		
		final RankingSample rankingSample = (RankingSample)sample;
		assert rankingSample.queryBoundaries.length -1 == rankingSample.numQueries;
		final double[] naturalOrder = computeNaturalOrderScores(predictions.length, rankingSample.queryBoundaries);
				
		double[] baselinePerQuery = ((RankingEvaluationMetric) parent).measureByQuery(naturalOrder, sample);
		double[] perQuery = ((RankingEvaluationMetric) parent).measureByQuery(predictions, sample);
		return T_measure(baselinePerQuery, perQuery, this.ALPHA);
	}
	
	@Override
	public SwapScorer getSwapScorer(double[] targets, int[] boundaries,
			int trunc, int[][] labelCounts) throws Exception 
	{
		final SwapScorer parentMeasure = ((RankingEvaluationMetric) parent).getSwapScorer(targets, boundaries, trunc, labelCounts);
		return new SAROSwapScorer(targets, boundaries, trunc, labelCounts,
				ALPHA, 
				parentMeasure);
	}
	
}
