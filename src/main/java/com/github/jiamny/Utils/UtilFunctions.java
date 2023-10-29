package com.github.jiamny.Utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import org.checkerframework.checker.units.qual.A;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

public class UtilFunctions {

    // Saved in the utils file for later use
    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle){
        return new ArrayDataset.Builder()
                .setData(features) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize,shuffle) // set the batch size and random sampling
                .build();
    }

    public static double[] ShuffleArray(double[] array) {
        Random rand = new Random();  // Random value generator
        for (int i = 0; i < array.length; i++) {
            int randomIndex = rand.nextInt(array.length);
            double temp = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = temp;
        }
        return array;
    }

    public static Object [] ShuffleArray(Object [] array) {
        Random rand = new Random();  // Random value generator
        for (int i = 0; i < array.length; i++) {
            int randomIndex = rand.nextInt(array.length);
            Object temp = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = temp;
        }
        return array;
    }

    public static int [] ShuffleArray(int [] array) {
        Random rand = new Random();  // Random value generator
        for (int i = 0; i < array.length; i++) {
            int randomIndex = rand.nextInt(array.length);
            int temp = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = temp;
        }
        return array;
    }

    public static ArrayList<NDArray> shuffle(NDArray X, NDArray y) {
        NDManager manager = NDManager.newBaseManager();
        int size = (int)X.getShape().get(0);
        int [] idx = new int[size];
        for(int i = 0; i < idx.length; i++)
            idx[i] = i;

        idx = ShuffleArray(idx);
        NDArray sIdx = manager.create(idx);
        ArrayList<NDArray> res = new ArrayList<>();
        res.add(X.get(sIdx));
        res.add(y.get(sIdx));
        return res;
    }

    public static ArrayList<NDArray> loadIrisData(String fileName) {
        // ----------------------------------------------
        HashMap<String, Integer> names = new HashMap<>();
        names.put("Iris-setosa", 0);
        names.put("Iris-versicolor", 1);
        names.put("Iris-virginica", 2);

        ArrayList<String> contents = new ArrayList<>();
        int ncol = 0;
        try {
            File fr = new File(fileName);
            BufferedReader in = null;

            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr)));
            String line = in.readLine();
            String[] curLine = line.strip().split(",");
            ncol = curLine.length;
            contents.add(line.strip());
            while( (line = in.readLine()) != null) {
                //System.out.println(line);
                contents.add(line.strip());
            }
            in.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        Object [] cts = ShuffleArray(contents.toArray());

        double [][] Xd = new double[cts.length][ncol - 1];
        int [][]  Yd  = new int[cts.length][1];

        for( int j = 0; j < cts.length; j++ ) {
            if (((String)cts[j]).length() < 1)
                continue;
            String [] curLine = ((String)cts[j]).strip().split(",");

            // skip
            for(int i = 0; i < (curLine.length - 1); i++) {
                Xd[j][i] = Double.parseDouble(curLine[i]);
            }
            //System.out.println(names.get(curLine[curLine.length - 1]));
            Yd[j][0] = names.get(curLine[curLine.length - 1]);
        }

        NDManager manager = NDManager.newBaseManager();
        ArrayList<NDArray> xy = new ArrayList<>();
        xy.add(manager.create(Xd).toType(DataType.FLOAT64, false));
        xy.add(manager.create(Yd).toType(DataType.INT32, false));
        return xy;
    }

    public static ArrayList<NDArray> train_test_split(NDArray X, NDArray y, double test_size) {
        NDManager manager = NDManager.newBaseManager();

        int num_rows = (int)(X.size(0));
        int num_test = (int)(num_rows * 0.2);
        Set<Integer> testIdx = new HashSet<>();
        int j = 0;
        while(j < num_test) {
            NDArray idx = manager.randomInteger(0, num_rows, new Shape(1), DataType.INT32);
            int i = (idx.toDevice(Device.cpu(), false)).toIntArray()[0];
            if( ! testIdx.contains(i) ) {
                testIdx.add(i);
                j++;
            }
        }
        int [] tidx = testIdx.stream().mapToInt(x->x).toArray();
        Arrays.sort(tidx);
        //Arrays.stream(tidx).forEach(System.out::println);

        ArrayList<Integer> tIdx = new ArrayList<>();
        for(int i = 0; i < num_rows; i++)
            if( ! testIdx.contains(i) )
                tIdx.add(i);
        int [] trainIdx = tIdx.stream().mapToInt(x -> x).toArray();
        //System.out.println(tidx.length + " " + trainIdx.length);

        NDArray tstIdx = manager.create(tidx).toType(DataType.INT32, false);
        NDArray trtIdx = manager.create(trainIdx).toType(DataType.INT32, false);

        ArrayList<NDArray> res = new ArrayList<>();
        res.add(X.get(trtIdx));
        res.add(y.get(trtIdx));
        res.add(X.get(tstIdx));
        res.add(y.get(tstIdx));
        return res;
    }

    public static ArrayList<NDArray> load_breast_cancer(String fName) {
        ArrayList<String> contents = new ArrayList<>();
        int ncol = 0;
        try {
            File fr = new File(fName);
            BufferedReader in = null;

            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr)));
            String line = in.readLine();  // skip column name line
            String[] curLine = line.strip().split(",");
            ncol = curLine.length;
            while( (line = in.readLine()) != null) {
                //System.out.println(line);
                contents.add(line.strip());
            }
            in.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        double [][] Xd = new double[contents.size()][ncol - 2];
        int [][]  Yd  = new int[contents.size()][1];

        for( int j = 0; j < contents.size(); j++ ) {
            if (contents.get(j).length() < 1) {
                Yd[j][0] = 1;
                continue;
            }
            String[] curLine = contents.get(j).strip().split(",");
            // skip
            for(int i = 2; i < curLine.length; i++) {
                Xd[j][i-2] = Double.parseDouble(curLine[i]);
            }
            if( curLine[1].equalsIgnoreCase("M") )
                Yd[j][0] = 0;
        }
        NDManager manager = NDManager.newBaseManager();
        ArrayList<NDArray> xy = new ArrayList<>();
        xy.add(manager.create(Xd).toType(DataType.FLOAT64, false));
        xy.add(manager.create(Yd).toType(DataType.INT32, false));
        return xy;
    }
}
