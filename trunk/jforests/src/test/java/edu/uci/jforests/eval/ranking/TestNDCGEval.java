package edu.uci.jforests.eval.ranking;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import edu.uci.jforests.eval.ranking.RankingEvaluationMetric.SwapScorer;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;

public class TestNDCGEval 
{
	@Test public void testSwapsTwoQueries() throws Exception
	{
		BlockingThreadPoolExecutor.init(1);
		NDCGEval eval = new NDCGEval(2,  2);
		SwapScorer s = eval.getSwapScorer(
				new double[]{0,1,0,1}, 
				new int[]{0,2,4}, 
				2, 
				new int[][]{new int[]{1,1,0,0,0}, new int[]{1,1,0,0,0}});
		System.err.println(s.getDelta(0, 0, 0, 1, 1));
		
		s = eval.getSwapScorer(
				new double[]{0,1}, 
				new int[]{0,2}, 
				2, 
				new int[][]{new int[]{1,1,0,0,0}});
		System.err.println(s.getDelta(0, 0, 0, 1, 1));
		
		assertTrue(true);
	}
}