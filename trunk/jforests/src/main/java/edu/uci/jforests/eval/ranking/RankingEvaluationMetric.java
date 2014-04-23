package edu.uci.jforests.eval.ranking;

import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.sample.Sample;

public abstract class RankingEvaluationMetric extends EvaluationMetric {

	
	/** This gives the ordering of docuuments if they remained 
	 * in their original order
	 * @param numTargets how many documents we want in the ranking
	 * @param boundaries the query offsets 
	 * @return an array with the ranking offsets for each query
	 */
	public static double[] computeNaturalOrderScores(int numTargets, int[] boundaries)
	{
		final double[] naturalOrder = new double[numTargets];
		for(int q=0;q<boundaries.length -1;q++)
		{
			int begin = boundaries[q];
			int numDocs = boundaries[q + 1] - begin;
			int nextScore = numDocs;
			for(int d=begin;d<(begin+numDocs);d++)
			{
				naturalOrder[d] = nextScore--;
			}
		}
		return naturalOrder;
	}
	
	public static abstract class SwapScorer {
		
		double[] targets;
		int[] boundaries;
		int trunc;
		int[][] labelCounts;
		
		/** 
		 * @param targets Targets of all documents in dataset. Equals Dataset.*.length
		 * @param boundaries Start and end indices of each query within the dataset.
		 * @param trunc Truncation level for evaluation measure
		 * @param labelCounts How many of each label value
		 */
		SwapScorer (double[] targets, int[] boundaries, int trunc, int[][] labelCounts)
		{
			this.targets = targets;
			this.boundaries = boundaries;
			this.trunc = trunc;
			this.labelCounts = labelCounts;
		}
		
		/** Calculates the difference in the measure in changing documents at rank i and rank j, which appear at 
		 * betterIdx and worseIdx in the dataset, within query queryIndex
		 * @param queryIndex the query offset
		 * @param betterIdx the offset of the better document in the dataset
		 * @param rank_i calculated rank for that query of the better document
		 * @param worseIdx the offset of the worse document in the dataset
		 * @param rank_j calculated rank for that query of the worse document
		 * @return This is NOT an absolute value. It is also the change in the per-query value, not in the MEAN over all queries.
		 */
		public abstract double getDelta(int queryIndex, int betterIdx, int rank_i, int worseIdx, int rank_j);
		
		public void setCurrentIterationEvaluation(int iteration, double[] nDCG) {}
		
		public int[] getQueryBoundaries()
		{
			return boundaries;
		}
		
	}

	public RankingEvaluationMetric(boolean isLargerBetter) {
		super(isLargerBetter);
	}


	/** 
	 * @param targets Targets of all documents in dataset. Equals Dataset.*.length
	 * @param boundaries Start and end indices of each query within the dataset.
	 * @param trunc Truncation level for evaluation measure
	 * @param labelCounts How many of each label value
	 * @throws Exception if things go pear shaped
	 */
	public abstract SwapScorer getSwapScorer(double[] targets, int[] boundaries, int trunc, int[][] labelCounts) throws Exception;
	
	public abstract double[] measureByQuery(double[] predictions, Sample sample) throws Exception;
	
	@Override
	public double measure(double[] predictions, Sample sample) throws Exception {
		final double[] result = measureByQuery(predictions, sample);
		double rtr = 0;
		for (int i = 0; i < result.length; i++) {
			rtr += result[i];			
		}
		rtr /= (double)result.length;		
		return rtr;
	}

	public EvaluationMetric getParentMetric() {
		return this;
	}

}
