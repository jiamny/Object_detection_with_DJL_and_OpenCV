package com.github.jiamny.Social_distance_monitoring;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static com.github.jiamny.Utils.HelperFunctions.combinationUtil;
import static com.github.jiamny.Utils.HelperFunctions.printVectorElements;
import static org.opencv.imgproc.Imgproc.line;

public class DeepSocial {
    public static ArrayList<Point> midPointCircleDraw(int x_centre, int y_centre, int r) {
        ArrayList<Point> points = new ArrayList<>();
        int x = r;
        int y = 0;
        points.add(new Point(x + x_centre, y + y_centre));
        if( r > 0 ) {
            points.add(new Point(x + x_centre, -y + y_centre));
            points.add(new Point(y + x_centre, x + y_centre));
            points.add(new Point(-1*y + x_centre, x + y_centre));
        }
        int P = 1 - r;
        while( x > y ) {
            y += 1;
            if( P <= 0 ) {
                P = P + 2 * y + 1;
            } else {
                x -= 1;
                P = P + 2 * y - 2 * x + 1;
            }
            if( x < y )
                break;
            points.add(new Point(x + x_centre, y + y_centre));
            points.add(new Point(-1*x + x_centre, y + y_centre));
            points.add(new Point(x + x_centre, -y + y_centre));
            points.add(new Point(-1*x + x_centre, -y + y_centre));
            if( x != y ) {
                points.add(new Point(y + x_centre, x + y_centre));
                points.add(new Point(-1*y + x_centre, x + y_centre));
                points.add(new Point(y + x_centre, -x + y_centre));
                points.add(new Point(-1*y + x_centre, -x + y_centre));
            }
        }
        return points;
    }

    public static boolean checkupArea(Mat img, double leftRange, double downRange, Point point, char clr, boolean Draw) {
        int hmax = img.rows() , wmax = img.cols();
        int hmin = hmax - (int)(hmax * downRange);
        int wmin = (int)(wmax * leftRange);
        if( Draw ) {
            Scalar color = new Scalar(0, 0, 255);
            if( clr == 'r' )
                color = new Scalar(0, 0, 255);
            if( clr == 'g' )
                color = new Scalar(0, 255, 0);
            if( clr == 'b' )
                color = new Scalar(255, 0, 0);
            if( clr == 'k' )
                color = new Scalar(0, 0, 0);
            line(img, new Point(0, hmin), new Point(wmax, hmin),color, 1);
            line(img, new Point(wmin, 0), new Point(wmin, hmax),color, 1);
        }
        double x = point.x, y = point.y;
        if( x < wmin )
            if( y > hmin )
                return true;
        return false;
    }

    public static double  Euclidean_distance(int [] p1, int[] p2) {
        return Math.sqrt(Math.pow((p1[0] - p2[0])*1.0, 2) + Math.pow((p1[1] - p2[1])*1.0, 2));
    }

    public static ArrayList<ArrayList<Integer>> find_zone(HashMap<Integer, List<Integer>> centroid_dict, int ViolationDistForIndivisuals) {
        ArrayList<Integer> redZone = new ArrayList<>();
        ArrayList<Integer>  greenZone = new ArrayList<>();

        ArrayList<ArrayList<Integer>> rlt = new ArrayList<>();

        int [] items = centroid_dict.keySet().stream().mapToInt(i->i).toArray();
        //printVectorElements(items);

        int r = 2;
        int n = items.length;
        int [] data = new int[r];
        ArrayList<List<Integer>> combLst = new ArrayList<>();
        combinationUtil(items, data,  0, n-1, 0, r, combLst);
        //System.out.println(combLst);

        for( List<Integer> itm : combLst ) {
            int[] pairk = itm.stream().mapToInt(i->i).toArray();
            int id1= pairk[0], id2 = pairk[1];

            int[] p1 = centroid_dict.get(Integer.valueOf(id1)).stream().mapToInt(i->i).toArray();
            int[] p2 = centroid_dict.get(Integer.valueOf(id2)).stream().mapToInt(i->i).toArray();

            double distance = Euclidean_distance(p1, p2);

            if( distance < ViolationDistForIndivisuals ) {
                if( ! redZone.contains(Integer.valueOf(id1)) )
                    redZone.add(id1);
                if( ! redZone.contains(Integer.valueOf(id2)) )
                    redZone.add(id2);
            }
        }
        for(int idx : items ) {
            if( ! redZone.contains(Integer.valueOf(idx)) )
                greenZone.add(idx);
        }
        rlt.add(redZone);
        rlt.add(greenZone);
        return rlt;
    }

}
