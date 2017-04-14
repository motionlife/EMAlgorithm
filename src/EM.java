import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class EM {
    private static String filepath = "dataset/em_data.txt";
    private static final int CLUSTER = 3;
    private static double[] miu = new double[CLUSTER];
    private static double[] segma = {1, 1, 1};
    private static double[] pi = new double[CLUSTER];

    public static void main(String[] args) {

    }

    private static ArrayList<Double> paseData(String file) {
        ArrayList<Double> data = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            while ((line = br.readLine()) != null) {
                data.add(Double.parseDouble(line));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static void EMAlgorithm(int itr, boolean fixVar, double[] X) {
        //1. Develop several randomized initialization strategies.

        int size = X.length;
        double[][] rsp = new double[size][CLUSTER];

        while(itr-->0) {
            //2. E Step -- Evaluate the responsibilities using current parameters
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) {
                    double diff = X[n] - miu[k];
                    rsp[n][k] = Math.exp(-diff * diff / (2 * segma[k])) / Math.sqrt(segma[k]);
                }
            }
            for (int n = 0; n < size; n++) {
                double denominator = 0;
                for (int k = 0; k < CLUSTER; k++) denominator += rsp[n][k];
                for (int k = 0; k < CLUSTER; k++) rsp[n][k] /= denominator;
            }

            //3. M Step -- Re-estimate the parameters using the current responsibilities.
            double[] N = new double[CLUSTER];
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) N[k] += rsp[n][k];
            }
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) miu[k] += rsp[n][k] * X[n];
                miu[k] /= N[k];
            }
            for (int k = 0; k < CLUSTER; k++) {
                for (int n = 0; n < size; n++) {
                    double diff = X[n] - miu[k];
                    segma[k] += rsp[n][k] * diff * diff;
                }
                segma[k] /= N[k];
            }
            for (int k = 0; k < CLUSTER; k++) pi[k] = N[k] / size;
        }
    }

}
