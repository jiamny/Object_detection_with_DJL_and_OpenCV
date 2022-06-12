import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.github.jiamny.Utils.Utils.combinationUtil;
import static com.github.jiamny.Utils.Utils.printVectorElements;

public class CombinationTest {

    @Test
    public void testCombination() {
        int arr[] = {1, 2, 3, 4, 5, 6};
        int r = 2;
        int n = arr.length;
        int data[] = new int[r];

        ArrayList<List<Integer>> lst = new ArrayList<>();
        lst.add(Arrays.stream(arr).boxed().collect(Collectors.toList()));
        printVectorElements(lst.get(0).stream().mapToInt(i->i).toArray());
        lst.clear();

        // Arrays.stream(ints).boxed().toList();
        combinationUtil(arr, data,  0, n-1, 0, r, lst);

        System.out.println("******************************************");
        for( List<Integer> e : lst ) {
            printVectorElements(e.stream().mapToInt(i->i).toArray());
        }

        for( int i : arr )
            System.out.println(i);
    }
}
