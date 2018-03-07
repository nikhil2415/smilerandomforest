package com.company;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import smile.classification.SVM;
import smile.math.kernel.LinearKernel;
import smile.math.kernel.GaussianKernel;
import smile.classification.RandomForest;
import smile.plot.Heatmap;
import smile.plot.PlotCanvas;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        // Reading training data
        MatFileReader mfr = new MatFileReader("SpamTrain.mat");
        Map<String, MLArray> content = mfr.getContent();
        double[][] X = ((MLDouble) content.get("X")).getArray();
        double[][] Y = ((MLDouble) content.get("y")).getArray();
        int[] y = new int[Y.length];
        for (int i = 0; i < Y.length; ++i)
            y[i] = (int) Y[i][0];

        // Reading test data
        MatFileReader mfrtest = new MatFileReader("SpamTest.mat");
        Map<String, MLArray> contenttest = mfrtest.getContent();
        double[][] Xtest = ((MLDouble) contenttest.get("Xtest")).getArray();
        double[][] Ytest = ((MLDouble) contenttest.get("ytest")).getArray();

        int[] ytest = new int[Ytest.length];
        for (int i = 0; i < Ytest.length; ++i)
            ytest[i] = (int) Ytest[i][0];

////////////////////////////Train models/////////////////////////////////////


        RandomForest rf = new RandomForest(X, y, 100);

        double sigma = 0.1;
        SVM<double[]> svmgauss = new SVM<double[]>(new GaussianKernel(sigma), 5);
        svmgauss.learn(X, y);
        svmgauss.finish();

        SVM<double[]> svmlinear = new SVM<double[]>(new LinearKernel(), 0.1);
        svmlinear.learn(X, y);
        svmlinear.finish();


////////////////////////////Predicting The Accuracy////////////////////////////////////

        int errorsvmgauss = 0;
        int errorsvmlinear = 0;
        int errorrf = 0;
        for (int i = 0; i < Xtest.length; i++) {
            if (svmgauss.predict(Xtest[i]) != ytest[i]) {
                errorsvmgauss++;
            }
            if (svmlinear.predict(Xtest[i]) != ytest[i]) {
                errorsvmlinear++;
            }
            if (rf.predict(Xtest[i]) != ytest[i]) {
                errorrf++;
            }
        }
        double accuracysvmgauss = ((Xtest.length-errorsvmgauss)*100)/Xtest.length;
        double accuracysvmlinear = ((Xtest.length-errorsvmlinear)*100)/Xtest.length;
        double accuracyrf=((Xtest.length-errorrf)*100)/Xtest.length;

        System.out.println("Accuracy of RandomForest with 100 trees: " + accuracyrf);
        System.out.println("Accuracy of SVM linear: " + accuracysvmlinear);
        System.out.println("Accuracy of SVM gauss, sigma = " + sigma+ ": " + accuracysvmgauss);

    }
}
