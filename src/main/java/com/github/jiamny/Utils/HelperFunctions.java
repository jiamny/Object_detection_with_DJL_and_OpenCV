package com.github.jiamny.Utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.*;
import java.util.stream.Collectors;

public class HelperFunctions {

    public static NDArray linspace(float min, float max, int size) {
        NDManager manager = NDManager.newBaseManager();
        float step = (max - min) / size;
        float[] data = new float[size + 1];
        int idx = 0;
        for (float i = min; i <= max; i += step)
            data[idx++] = i;

        return manager.create(data);
    }

    public static ArrayList<NDArray> meshgrid(NDArray A, NDArray B) {
        ArrayList<NDArray> dt = new ArrayList<>();

        NDArray As = A.duplicate().reshape(1, -1);
        NDArray Bs = B.duplicate().reshape(-1, 1);

        for (int i = 1; i < A.size(); i++)
            As = As.concat(A.duplicate().reshape(1, -1), 0);

        dt.add(As);

        for (int i = 1; i < B.size(); i++)
            Bs = Bs.concat(B.duplicate().reshape(-1, 1), 1);

        dt.add(Bs);
        return dt;
    }

    public static <T extends Number & Comparable<T>> double median(Collection<T> numbers) {
        if (numbers.isEmpty()) {
            throw new IllegalArgumentException("Cannot compute median on empty collection of numbers");
        }
        List<T> numbersList = new ArrayList<>(numbers);
        Collections.sort(numbersList);
        int middle = numbersList.size() / 2;
        if (numbersList.size() % 2 == 0) {
            return 0.5 * (numbersList.get(middle).doubleValue() + numbersList.get(middle - 1).doubleValue());
        } else {
            return numbersList.get(middle).doubleValue();
        }
    }

    public static void printVectorElements(double[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorElements(int[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorElements(long[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorObjects(Object[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printListObjects(List<Object> ax) {
        ax.forEach(System.out::println);
        System.out.println();
    }

    public static ArrayList<Double> cumulativeSum(ArrayList<Double> numbers) {
        // variable
        double sum = 0.0;

        // traverse through the array
        for (int i = 0; i < numbers.size(); i++) {
            sum += numbers.get(i); // find sum
            numbers.set(i, sum);   // replace
        }
        // return
        return numbers;
    }

    public static <K, V> Map.Entry<K, V> min(Map<K, V> map, Comparator<V> comp) {
        Iterator<Map.Entry<K, V>> entries = map.entrySet().iterator();

        if (!entries.hasNext()) {
            return null;
        }
        Map.Entry<K, V> min;
        for (min = entries.next(); entries.hasNext(); ) {
            Map.Entry<K, V> value = entries.next();
            if (comp.compare(value.getValue(), min.getValue()) < 0) {
                min = value;
            }
        }
        return min;
    }

    public static HashMap<String, Double> sortByValueJava8Stream(HashMap<String, Double> unSortedMap, boolean reverse) {

        HashMap<String, Double> sortedMap = new HashMap<>();
        if (reverse) {
            unSortedMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .forEachOrdered(x -> sortedMap.put(x.getKey(), x.getValue()));
            System.out.println("Reverse Sorted Map   : " + sortedMap);
        } else {
            unSortedMap.entrySet().stream().sorted(Map.Entry.comparingByValue())
                    .forEachOrdered(x -> sortedMap.put(x.getKey(), x.getValue()));
            System.out.println("Sorted Map   : " + sortedMap);
        }
        return sortedMap;
    }

    /* arr[]  ---> Input Array
    data[] ---> Temporary array to store current combination
    start & end ---> Starting and Ending indexes in arr[]
    index  ---> Current index in data[]
    r ---> Size of a combination to be printed */
    public static void combinationUtil(int arr[], int data[], int start,
                                       int end, int index, int r, ArrayList<List<Integer>> lst) {
        // Current combination is ready to be printed, print it
        if (index == r) {
            //for (int j=0; j<r; j++)
            //    System.out.print(data[j]+" ");
            //System.out.println("");
            lst.add(Arrays.stream(data).boxed().collect(Collectors.toList()));
            return;
        }

        // replace index with all possible elements. The condition
        // "end-i+1 >= r-index" makes sure that including one element
        // at index will make a combination with remaining elements
        // at remaining positions
        for (int i = start; i <= end && end - i + 1 >= r - index; i++) {
            data[index] = arr[i];
            combinationUtil(arr, data, i + 1, end, index + 1, r, lst);
        }
    }

    public static int [] range(int start, int end) {
        int size = (end - start);
        int [] rg = new int[size];
        for( int i = start; i < end; i++ )
            rg[i-start] = i;
        return rg;
    }

    public static HashMap<Integer, Double> sortByValue(HashMap<Integer, Double> hm, boolean des) {
        // Create a list from elements of HashMap
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double> >() {
            public int compare(Map.Entry<Integer, Double> o1,
                               Map.Entry<Integer, Double> o2) {
                if( des )
                    return (o2.getValue()).compareTo(o1.getValue());
                else
                    return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<Integer, Double> temp = new LinkedHashMap<Integer, Double>();
        for (Map.Entry<Integer, Double> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }

    public static int [] argSort(double [] d) {
        HashMap<Integer, Double> hm = new HashMap<>();
        for(int i = 0; i < d.length; i++) {
            hm.put(i, d[i]);
        }
        hm = sortByValue(hm, true);
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());
        int [] idx = new int[d.length];
        int i = 0;
        for (Map.Entry<Integer, Double> aa : list) {
            idx[i] = aa.getKey();
            i++;
        }
        return idx;
    }

    // ArrayList to Array Conversion
    public static int[] toIntArray(ArrayList<Integer> al) {
        return al.stream().mapToInt(i -> i).toArray();
    }

    public static double[] toDoubleArray(ArrayList<Double> al) {
        return al.stream().mapToDouble(i -> i).toArray();
    }
}
