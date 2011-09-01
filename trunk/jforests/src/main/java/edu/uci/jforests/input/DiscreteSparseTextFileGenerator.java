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

package edu.uci.jforests.input;

import java.io.File;

import java.io.PrintStream;

import edu.uci.jforests.input.sparse.SparseTextFileLine;
import edu.uci.jforests.input.sparse.SparseTextFileReader;

/**
 * Converts continuous feature values to 2-byte integer values
 */

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class DiscreteSparseTextFileGenerator {

	public static void convert(String inputFilename, String featuresStatFile, String outputFilename) {
		
		if (new File(outputFilename).exists()) {
			System.out.println("File: " + outputFilename + " already exists. Skipping it.");
			return;
		}
		
		FeatureAnalyzer featureAnalyzer = new FeatureAnalyzer();
		featureAnalyzer.loadFeaturesFromFile(featuresStatFile);

		StringBuilder sb = new StringBuilder();
		double value;
		SparseTextFileReader reader = new SparseTextFileReader();
		SparseTextFileLine line = new SparseTextFileLine();
		reader.open(inputFilename);
		int intValue;
		try {
			PrintStream output = new PrintStream(new File(outputFilename));
			int count = 0;
			while (reader.loadNextLine(line)) {
				if (line.meta) {
					output.println(line.content);
					continue;
				}
				sb.setLength(0);
				sb.append(line.target);
				if (line.qid != null) {
					sb.append(" qid:" + line.qid);
				}
				for (int i = 0; i < line.numPairs; i++) {
					FeatureValuePair pair = line.pairs[i];
					value = pair.featureValue;
					int idx = pair.featureIndex - 1;
					if (featureAnalyzer.onLogScale[idx]) {
						value = (Math.log(value - featureAnalyzer.min[idx] + 1) * featureAnalyzer.factor[idx]);
					} else {
						value = (value - featureAnalyzer.min[idx]) * featureAnalyzer.factor[idx];
					}
					intValue = (int) Math.round(value);
					if (intValue != 0) {
						sb.append(" " + pair.featureIndex + ":" + intValue);
					}
				}
				output.println(sb.toString());
				count++;
				if (count % 10000 == 0) {
					System.out.println(count);
				}
			}
			output.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		reader.close();
	}
}
