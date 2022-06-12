package com.github.jiamny.Social_distance_monitoring;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import java.util.ArrayList;

import static com.github.jiamny.Social_distance_monitoring.DeepSocial.midPointCircleDraw;
import static org.opencv.core.Core.perspectiveTransform;
import static org.opencv.imgproc.Imgproc.getPerspectiveTransform;
import static org.opencv.imgproc.Imgproc.warpPerspective;

public class BirdsEye {
    private int c = 0, r = 0;
    private Mat transferI2B = new Mat(), transferB2I = new Mat();

    public Mat image = new Mat();
    public Mat original = new Mat();
    public Mat bird = new Mat();
    public BirdsEye(Mat image, MatOfPoint2f cordinates) {
        this.original = image.clone();
        this.image = image;
        this.c = image.size(0);
        this.r = image.size(1);
        MatOfPoint2f pst2 = cordinates; //np.float32(cordinates)
        MatOfPoint2f pst1 = new MatOfPoint2f(
                new Point(0.0, 0.0),
                new Point(this.r * 1.0, 0.0),
                new Point(0.0, this.c * 1.0),
                new Point(this.r * 1.0, this.c * 1.0)
        );
        //np.float32([[0, 0], [this.r, 0], [0, this.c], [this.r, this.c]]);
        this.transferI2B = getPerspectiveTransform(pst1, pst2);
        this.transferB2I = getPerspectiveTransform(pst2, pst1);
        this.img2bird();
    }

    public Mat img2bird() {
        warpPerspective(this.image, this.bird, this.transferI2B, new Size(this.r, this.c));
        return this.bird;
    }

    public Mat bird2img() {
        warpPerspective(this.bird, this.image, this.transferB2I, new Size(this.r, this.c));
        return this.image;
    }

    public void setImage(Mat img) {
        this.image = img;
    }

    public void setBird(Mat bird) {
        this.bird = bird;
    }

    public Mat convrt2Bird(Mat img) {
        warpPerspective(img, this.bird, this.transferI2B, new Size(this.r, this.c));
        return this.bird;
    }

    public Mat convrt2Image(Mat bird) {
        warpPerspective(bird, this.image, this.transferB2I, new Size(this.r, this.c));
        return this.image;
    }

    public Point projection_on_bird(Point p) {
        Mat M = this.transferI2B;
        // px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        double px = (M.get(0,0)[0] * p.x + M.get(0,1)[0] * p.y + M.get(0,2)[0])*1.0 /
                    (M.get(2,0)[0] * p.x + M.get(2,1)[0] * p.y + M.get(2,2)[0]);
        double py = (M.get(1,0)[0] * p.x + M.get(1,1)[0] * p.y + M.get(1,2)[0])*1.0 /
                    (M.get(2,0)[0] * p.x + M.get(2,1)[0] * p.y + M.get(2,2)[0]);
        return new Point((int)px, (int)py);
    }

    public Point projection_on_image(Point p) {
        Mat M = this.transferB2I;
        double px = (M.get(0,0)[0] * p.x + M.get(0,1)[0] * p.y + M.get(0,2)[0])*1.0 /
                    (M.get(2,0)[0] * p.x + M.get(2,1)[0] * p.y + M.get(2,2)[0]);
        double py = (M.get(1,0)[0] * p.x + M.get(1,1)[0] * p.y + M.get(1,2)[0])*1.0 /
                    (M.get(2,0)[0] * p.x + M.get(2,1)[0] * p.y + M.get(2,2)[0]);
        return new Point((int)px, (int)py);
    }

    ArrayList<Point> points_projection_on_image(Point center, int radius) {
        int x = (int)(center.x), y = (int)(center.y);
        ArrayList<Point> points = midPointCircleDraw(x, y, radius);
        Mat transformed = new Mat();

        Mat original = Converters.vector_Point2f_to_Mat(points);
        perspectiveTransform(original, transformed, this.transferB2I);
        ArrayList<Point> transPoints = new ArrayList<Point>();
        Converters.Mat_to_vector_Point2f(transformed, transPoints);
        return transPoints;
    }
}
