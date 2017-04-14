import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class EM {
    private static final int CLUSTER = 3;
    private static double[] miu = new double[CLUSTER];
    private static double[] sigma = {1, 1, 1};
    private static double[] pi = new double[CLUSTER];

    public static void main(String[] args) {
        double[] data = parseData().stream().mapToDouble(Double::doubleValue).toArray();
        init(false, data);
        EMAlgorithm(300, false, data);
        Arrays.stream(miu).forEach(System.out::println);
        Arrays.stream(sigma).forEach(System.out::println);
        Arrays.stream(pi).forEach(System.out::println);

        System.out.println("\n"+logLikelihood(data));
    }

    /**
     * Read data from file
     */
    private static ArrayList<Double> parseData() {
        ArrayList<Double> data = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("dataset/em_data.txt"));
            String line;
            while ((line = br.readLine()) != null) {
                data.add(Double.parseDouble(line));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    /**
     * Get log likelihood
     */
    private static double logLikelihood(double[] X) {
        double llh = 0;
        for (double x : X) {
            double tmp = 0;
            for (int k = 0; k < CLUSTER; k++) {
                double diff = x - miu[k];
                tmp += pi[k] * Math.exp(-diff * diff / (2 * sigma[k])) / Math.sqrt(2 * Math.PI * sigma[k]);
            }
            llh += Math.log(tmp);
        }
        return llh;
    }

    /**
     * EM algorithm with Gaussian mixture model on 1-d data
     */
    private static void EMAlgorithm(int itr, boolean fixVar, double[] X) {
        int size = X.length;
        double[][] rsp = new double[size][CLUSTER];
        //1. Initialize parameters
        init(fixVar, X);

        while (itr-- > 0) {
            //2. E Step -- Evaluate the responsibilities using current parameters
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) {
                    double diff = X[n] - miu[k];
                    rsp[n][k] = pi[k] * Math.exp(-diff * diff / (2 * sigma[k])) / Math.sqrt(sigma[k]);
                }
            }
            for (int n = 0; n < size; n++) {
                double denominator = 0;
                for (int k = 0; k < CLUSTER; k++) denominator += rsp[n][k];
                for (int k = 0; k < CLUSTER; k++) rsp[n][k] /= denominator;
            }

            //3. M Step -- Re-estimate the parameters using the current responsibilities.
            //Calculate N[k]
            double[] N = new double[CLUSTER];
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) N[k] += rsp[n][k];
            }
            //Calculate the mean for each cluster component
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) miu[k] += rsp[n][k] * X[n];
                miu[k] /= N[k];
            }
            //Calculate the covariance for each cluster component
            if (!fixVar) {
                for (int k = 0; k < CLUSTER; k++) {
                    for (int n = 0; n < size; n++) {
                        double diff = X[n] - miu[k];
                        sigma[k] += rsp[n][k] * diff * diff;
                    }
                    sigma[k] /= N[k];
                }
            }
            //Calculate the mixing coefficient for each cluster component
            for (int k = 0; k < CLUSTER; k++) pi[k] = N[k] / size;
        }
    }

    /***
     * Develop several randomized initialization strategies
     */
    private static void init(boolean fixVar, double[] X) {
        for (int k = 0; k < CLUSTER; k++) miu[k] = X[new Random().nextInt(X.length)] * 0.777;
        if (!fixVar) {
            for (int k = 0; k < CLUSTER; k++) {
                final double mean = miu[k];
                sigma[k] = Arrays.stream(X).map(d -> (d - mean) * (d - mean)).average().orElse(1);
            }
        }
        for (int k = 0; k < CLUSTER; k++) pi[k] = new Random().nextDouble();
        double sum = Arrays.stream(pi).sum();
        for (int k = 0; k < CLUSTER; k++) pi[k] /= sum;
    }

}
