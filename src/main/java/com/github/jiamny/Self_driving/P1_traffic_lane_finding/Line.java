package com.github.jiamny.Self_driving.P1_traffic_lane_finding;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Line {
    private double x1 = 0., y1 = 0., x2 = 0., y2 = 0.;
    private static final double eps = 2.220446049250313e-16;

    public double slope = 0.0;
    public double bias = 0.0;

    /*
    A Line is defined from two points (x1, y1) and (x2, y2) as follows:
    y - y1 = (y2 - y1) / (x2 - x1) * (x - x1)
    Each line has its own slope and intercept (bias).
     */
    public Line() {
    }

    public Line(double x1, double y1, double x2, double y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;

        this.slope = this.compute_slope();
        this.bias = this.compute_bias();
    }

    public double compute_slope() {
        return ((this.y2 - this.y1) / (this.x2 - this.x1 + eps));
    }

    public double compute_bias() {
        return (this.y1 - this.slope * this.x1);
    }

    public double[] get_coords() {
        double[] a = {this.x1, this.y1, this.x2, this.y2};
        return (a);
    }

    public void set_coords(double x1, double y1, double x2, double y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.slope = this.compute_slope();
        this.bias = this.compute_bias();
    }

    public void draw(Mat img, Scalar color, int thickness) {
        //has to be integer
        if (color == null) color = new Scalar(255, 0, 0);
        if (thickness <= 0) thickness = 10;
        //System.out.println( this.x1 + " " + this.y1 + " " + this.x2 + " " + this.y2);
        Imgproc.line(img, new Point((int) this.x1, (int) this.y1), new Point((int) this.x2, (int) this.y2), color, thickness);
    }
}