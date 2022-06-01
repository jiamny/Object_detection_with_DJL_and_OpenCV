package com.github.jiamny.Utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

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
}
