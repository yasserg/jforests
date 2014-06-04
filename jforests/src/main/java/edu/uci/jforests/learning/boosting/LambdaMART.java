/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.uci.jforests.learning.boosting;

import java.util.Arrays;

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.eval.ranking.NDCGEval;
import edu.uci.jforests.eval.ranking.RankingEvaluationMetric;
import edu.uci.jforests.learning.trees.LeafInstances;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.TreeLeafInstances;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.ArraysUtil;
import edu.uci.jforests.util.ConfigHolder;
import edu.uci.jforests.util.Constants;
import edu.uci.jforests.util.ScoreBasedComparator;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;
import edu.uci.jforests.util.concurrency.TaskCollection;
import edu.uci.jforests.util.concurrency.TaskItem;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class LambdaMART extends GradientBoosting {

	
	private TaskCollection<LambdaWorker> workers;

	private RankingEvaluationMetric.SwapScorer swapScorer;
	private double sigmoidParam;
	private double[] sigmoidCache;
	private double minScore;
	private double maxScore;
	private double sigmoidBinWidth;

	protected double[] denomWeights;
	
	private int[] subLearnerSampleIndicesInTrainSet;

	public LambdaMART() {
		super("LambdaMART");
	}

	public void init(ConfigHolder configHolder, RankingDataset dataset, int maxNumTrainInstances, int maxNumValidInstances, EvaluationMetric evaluationMetric)
			throws Exception {
		super.init(configHolder, maxNumTrainInstances, maxNumValidInstances, evaluationMetric);

		LambdaMARTConfig lambdaMartConfig = configHolder.getConfig(LambdaMARTConfig.class);
		GradientBoostingConfig gradientBoostingConfig = configHolder.getConfig(GradientBoostingConfig.class);
		int[][] labelCountsPerQuery = NDCGEval.getLabelCountsForQueries(dataset.targets, dataset.queryBoundaries);
		
		//instantiate the swap scorer. this measures the gradients for different swaps
		swapScorer = ((RankingEvaluationMetric) evaluationMetric).getSwapScorer(dataset.targets, dataset.queryBoundaries, lambdaMartConfig.maxDCGTruncation, labelCountsPerQuery);

		
		
		// Sigmoid parameter is set to be equal to the learning rate.
		sigmoidParam = gradientBoostingConfig.learningRate;

		initSigmoidCache(lambdaMartConfig.sigmoidBins, lambdaMartConfig.costFunction);

		workers = new TaskCollection<LambdaWorker>();
		int numWorkers = BlockingThreadPoolExecutor.getInstance().getMaximumPoolSize();
		for (int i = 0; i < numWorkers; i++) {
			workers.addTask(new LambdaWorker(dataset.maxDocsPerQuery));
		}

		denomWeights = new double[maxNumTrainInstances];
		subLearnerSampleIndicesInTrainSet = new int[maxNumTrainInstances];
	}

	private void initSigmoidCache(int sigmoidBins, String costFunction) throws Exception {
		minScore = Constants.MIN_EXP_POWER / sigmoidParam;
		maxScore = -minScore;

		sigmoidCache = new double[sigmoidBins];
		sigmoidBinWidth = (maxScore - minScore) / sigmoidBins;
		if (costFunction.equals("cross-entropy")) {
			double score;
			for (int i = 0; i < sigmoidBins; i++) {
				score = minScore + i * sigmoidBinWidth;
				if (score > 0.0) {
					sigmoidCache[i] = 1.0 - 1.0 / (1.0 + Math.exp(-sigmoidParam * score));
				} else {
					sigmoidCache[i] = 1.0 / (1.0 + Math.exp(sigmoidParam * score));
				}
			}
		} else if (costFunction.equals("fidelity")) {
			double score;
			for (int i = 0; i < sigmoidBins; i++) {
				score = minScore + i * sigmoidBinWidth;
				if (score > 0.0) {
					double exp = Math.exp(-2 * sigmoidParam * score);
					sigmoidCache[i] = (-sigmoidParam / 2) * Math.sqrt(exp / Math.pow(1 + exp, 3));
				} else {
					double exp = Math.exp(sigmoidParam * score);
					sigmoidCache[i] = (-sigmoidParam / 2) * Math.sqrt(exp / Math.pow(1 + exp, 3));
				}
			}
		} else {
			throw new Exception("Unknown cost function: " + costFunction);
		}
	}

	@Override
	protected void preprocess() {
		Arrays.fill(trainPredictions, 0, curTrainSet.size, 0);
		if (curValidSet != null)
			Arrays.fill(validPredictions, 0, curValidSet.size, 0);
		
		//calculate the effectiveness of the natural ranking. this is needed for U_risk
		
		RankingEvaluationMetric rankingMetric = (RankingEvaluationMetric) ( (RankingEvaluationMetric) evaluationMetric).getParentMetric();
		
		double[] nDCG = null;
		try {
			nDCG = ((RankingSample) curTrainSet).evaluateByQuery(
				RankingEvaluationMetric.computeNaturalOrderScores(curTrainSet.size, swapScorer.getQueryBoundaries()), 
				rankingMetric);			
		} catch (Exception e) {
			e.printStackTrace();
		}		
		swapScorer.setCurrentIterationEvaluation(0, nDCG);
	}

	@Override
	protected void postProcessScores() {
		// Do nothing
	}

	protected double getAdjustedOutput(LeafInstances leafInstances) {
		double numerator = 0.0;
		double denomerator = 0.0;
		int instance;
		for (int i = leafInstances.begin; i < leafInstances.end; i++) {
			instance = subLearnerSampleIndicesInTrainSet[leafInstances.indices[i]];
			numerator += residuals[instance];
			denomerator += denomWeights[instance];
		}
		return (numerator + Constants.EPSILON) / (denomerator + Constants.EPSILON);
	}

	@Override
	protected void adjustOutputs(Tree tree, TreeLeafInstances treeLeafInstances) {
		LeafInstances leafInstances = new LeafInstances();
		for (int l = 0; l < tree.numLeaves; l++) {
			treeLeafInstances.loadLeafInstances(l, leafInstances);
			double adjustedOutput = getAdjustedOutput(leafInstances);
			((RegressionTree) tree).setLeafOutput(l, adjustedOutput);
		}
	}

	protected void setSubLearnerSampleWeights(RankingSample sample) {
		// Do nothing (weights are equal to 1 in LambdaMART)
	}

	@Override
	protected Sample getSubLearnerSample() {
		Arrays.fill(residuals, 0, curTrainSet.size, 0);
		Arrays.fill(denomWeights, 0, curTrainSet.size, 0);
		RankingSample trainSample = (RankingSample) curTrainSet;
		int chunkSize = 1 + (trainSample.numQueries / workers.getSize());
		int offset = 0;
		for (int i = 0; i < workers.getSize() && offset < trainSample.numQueries; i++) {
			int endOffset = offset + Math.min(trainSample.numQueries - offset, chunkSize);
			workers.getTask(i).init(offset, endOffset);
			BlockingThreadPoolExecutor.getInstance().execute(workers.getTask(i));
			offset += chunkSize;
		}
		BlockingThreadPoolExecutor.getInstance().await();

		trainSample = trainSample.getClone();
		trainSample.targets = residuals;
		setSubLearnerSampleWeights(trainSample);

		RankingSample zeroFilteredSample = trainSample.getClone();
		RankingSample subLearnerSample = zeroFilteredSample.getRandomSubSample(samplingRate, rnd);
		for (int i = 0; i < subLearnerSample.size; i++) {
			subLearnerSampleIndicesInTrainSet[i] = zeroFilteredSample.indicesInParentSample[subLearnerSample.indicesInParentSample[i]];
		}
		return subLearnerSample;
	}

	@Override
	protected void onIterationEnd() {
		
		RankingEvaluationMetric rankingMetric = (RankingEvaluationMetric) ( (RankingEvaluationMetric) evaluationMetric).getParentMetric();
		
		//inform the swap scorer of the new training measurement
		double[] nDCG = null;
		try {
			nDCG = ((RankingSample) curTrainSet).evaluateByQuery(
				trainPredictions, rankingMetric);
		} catch (Exception e) {
			e.printStackTrace();
		}		
		swapScorer.setCurrentIterationEvaluation(curIteration, nDCG);
		
		
		super.onIterationEnd();
	}

	private class LambdaWorker extends TaskItem {

		private int[] permutation;
		private int beginIdx;
		private int endIdx;
		private ScoreBasedComparator comparator;

		public LambdaWorker(int maxDocsPerQuery) {
			permutation = new int[maxDocsPerQuery];
			comparator = new ScoreBasedComparator();
		}

		public void init(int beginIdx, int endIdx) {
			this.beginIdx = beginIdx;
			this.endIdx = endIdx;
			comparator.labels = curTrainSet.targets;
		}

		@Override
		public void run() {
			double scoreDiff;
			double rho;
			double deltaWeight;
			double pairWeight;
			RankingSample trainSet = (RankingSample) curTrainSet;
			double[] targets = trainSet.targets;
			comparator.scores = trainPredictions;
			try {
				for (int query = beginIdx; query < endIdx; query++) {
					int begin = trainSet.queryBoundaries[query];
					int numDocuments = trainSet.queryBoundaries[query + 1] - begin;

					//sort documents for this query by descending prediction (known as a permutation), recording previous position
					comparator.offset = begin;
					for (int d = 0; d < numDocuments; d++) {
						permutation[d] = d;
					}					
					ArraysUtil.insertionSort(permutation, numDocuments, comparator);
					//now permutations contains the offset of documents by rank
					//i.e. permutations[0] is the offset of the FIRST(??) document in trainSet
					
					//for each document for this query
					for (int i = 0; i < numDocuments; i++) {
						int betterIdx = permutation[i];
						if (targets[begin+betterIdx] > 0) {
							for (int j = 0; j < numDocuments; j++) {
								if (i != j) {
									int worseIdx = permutation[j];
									//if i should have been ranked above j
									if (targets[begin+betterIdx] > targets[begin+worseIdx]) {
										scoreDiff = trainPredictions[begin + betterIdx] - trainPredictions[begin + worseIdx];

										//calculate the original gradient (\lambda_{ij} according to Wang et al)
										if (scoreDiff <= minScore) {
											rho = sigmoidCache[0];
										} else if (scoreDiff >= maxScore) {
											rho = sigmoidCache[sigmoidCache.length - 1];
										} else {
											rho = sigmoidCache[(int) ((scoreDiff - minScore) / sigmoidBinWidth)];
										}

										//what would |delta M_{ij}| have been?
										pairWeight = Math.abs(
												swapScorer.getDelta(trainSet.queryIndices[query], begin+betterIdx, i, begin+worseIdx, j)
												);
										//System.err.printf(this.toString() + " query=%d betterIdx=%d worseIdx=%d i=%d j=%d pairWeight=%f\n", query, betterIdx, worseIdx, i, j, pairWeight);
										
										residuals[begin + betterIdx] += rho * pairWeight;
										residuals[begin + worseIdx] -= rho * pairWeight;

										deltaWeight = rho * (1.0 - rho) * pairWeight;
										denomWeights[begin + betterIdx] += deltaWeight;
										denomWeights[begin + worseIdx] += deltaWeight;
									}
								}
							}
						}
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
