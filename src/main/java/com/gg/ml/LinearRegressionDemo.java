package com.gg.ml;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
public class LinearRegressionDemo {

	public static final String TRAINING_DATA_SET_FILENAME="linear-train.arff";
	public static final String TESTING_DATA_SET_FILENAME="linear-test.arff";
	public static final String PREDICTION_DATA_SET_FILENAME="test-confused.arff";
	public static Instances getDataSet(String fileName) throws IOException {

		int classIdx = 1;
		ArffLoader loader = new ArffLoader();
		loader.setSource(LinearRegressionDemo.class.getResourceAsStream("/" + fileName));
		Instances dataSet = loader.getDataSet();
		dataSet.setClassIndex(classIdx);
		return dataSet;
	}

	public static void process() throws Exception {

		Instances trainingDataSet = getDataSet(TRAINING_DATA_SET_FILENAME);
		Instances testingDataSet = getDataSet(TESTING_DATA_SET_FILENAME);
		Classifier classifier = new weka.classifiers.functions.LinearRegression();
		classifier.buildClassifier(trainingDataSet);
		Evaluation eval = new Evaluation(trainingDataSet);
		eval.evaluateModel(classifier, testingDataSet);
		System.out.println("** Linear Regression Evaluation with Datasets **");
		System.out.println(eval.toSummaryString());
		System.out.print(" the expression for the input data as per alogorithm is ");
		System.out.println(classifier);

		Instance predicationDataSet = getDataSet(PREDICTION_DATA_SET_FILENAME).lastInstance();
		double value = classifier.classifyInstance(predicationDataSet);

		System.out.println(value);
	}
}
