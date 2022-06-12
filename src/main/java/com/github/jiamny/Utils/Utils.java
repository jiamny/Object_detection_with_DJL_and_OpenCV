package com.github.jiamny.Utils;

import java.util.*;
import java.util.stream.Collectors;

public class Utils {
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

    public static void printVectorElements(double [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorElements(int [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorObjects(Object [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
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
        for (min = entries.next(); entries.hasNext();) {
            Map.Entry<K, V> value = entries.next();
            if (comp.compare(value.getValue(), min.getValue()) < 0) {
                min = value;
            }
        }
        return min;
    }

    public static HashMap<String, Double> sortByValueJava8Stream( HashMap<String, Double> unSortedMap, boolean reverse) {

        HashMap<String, Double> sortedMap = new HashMap<>();
        if( reverse ) {
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
                                int end, int index, int r, ArrayList<List<Integer>> lst)
    {
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
        for (int i=start; i<=end && end-i+1 >= r-index; i++) {
            data[index] = arr[i];
            combinationUtil(arr, data, i+1, end, index+1, r, lst);
        }
    }

    // ArrayList to Array Conversion
    public static int [] toIntArray(ArrayList<Integer> al) {
      return  al.stream().mapToInt(i -> i).toArray();
    }

    public static double [] toDoubleArray(ArrayList<Double> al) {
        return  al.stream().mapToDouble(i -> i).toArray();
    }
}
