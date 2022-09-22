import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.github.jiamny.Self_driving.P1_traffic_lane_finding.LaneDetection;
import com.github.jiamny.Self_driving.P1_traffic_lane_finding.Line;
import com.github.jiamny.Utils.ImageViewer;
import com.github.jiamny.Utils.MaskMats;
import com.github.jiamny.Utils.Utils;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import static com.github.jiamny.Utils.ImageHelper.mat2DjlImage;
import static com.github.jiamny.Utils.ImageHelper.ndarrayToMat;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.*;

public class LaneFindingTest {
    @Test
    public void testLaneFinding() {
        System.load("/usr/local/share/java/opencv4/libopencv_java460.so");
        LaneDetection Ldt = new LaneDetection();

        boolean solid_lines = true;
        NDManager manager = NDManager.newBaseManager();
        String img_path = "./data/self_driving/images";
        Mat color_image = imread(img_path + "/solidYellowCurve2.jpg", IMREAD_COLOR);
        cvtColor(color_image, color_image, COLOR_BGR2RGB);

        // resize to 960 x 540
        resize(color_image, color_image, new Size(960, 540));

        // convert to grayscale
        Mat img_gray = new Mat();
        cvtColor(color_image, img_gray, COLOR_BGR2GRAY);

        // perform gaussian blur
        Mat img_blur = new Mat();
        GaussianBlur(img_gray, img_blur, new Size(17, 17), 0);

        // perform edge detection
        Mat img_edge = new Mat();
        Canny(img_blur, img_edge, 50., 80.);

        // perform hough transform
        // Mat detected_lines = hough_lines_detection(img_edge, 2, Math.PI / 180, 1, 15, 5);
        double rho = 2, theta = Math.PI / 180;
        int threshold = 1;
        double min_line_len = 15, max_line_gap = 5;

        //`img` should be the output of a Canny transform.
        Mat lines = Ldt.hough_lines_detection(img_edge, rho, theta, threshold, min_line_len, max_line_gap);
        //Imgproc.HoughLinesP(img_edge, lines, rho, theta, threshold, min_line_len, max_line_gap);

        System.out.println(lines.size() + " " + lines.rows() + " " + lines.cols());

        ArrayList<Line> d_lines = new ArrayList<>();

        for (int i = 0; i < lines.rows(); i++) {
            double[] val = lines.get(i, 0);
     //       System.out.println(val[0] + " " + val[1] + " " + val[2] + " " + val[3]);
            d_lines.add(new Line(val[0], val[1], val[2], val[3]));
        }

        ArrayList<Line> lane_lines = new ArrayList<>();
        ArrayList<Line> candidate_lines = new ArrayList<>();
        // if 'solid_lines' infer the two lane lines
        if( solid_lines ) {
            for( Line line : d_lines )
                // consider only lines with slope between 30 and 60 degrees
                if( (0.5 <= Math.abs(line.slope)) && (Math.abs(line.slope) <= 2) ) {
                    System.out.println(line.get_coords()[0] + " " + line.get_coords()[1] + " " + line.get_coords()[2] + " " + line.get_coords()[3]);
                    candidate_lines.add(line);
                }
            System.out.println(candidate_lines.size());
            // interpolate lines candidates to find both lanes
            //lane_lines = Ldt.compute_lane_from_candidates(candidate_lines, mat2DjlImage(img_gray).toNDArray(manager).getShape().getShape());
        } else {
            // if not solid_lines, just return the hough transform output
            lane_lines = d_lines;
        }

        long[] img_shape = mat2DjlImage(img_gray).toNDArray(manager).getShape().getShape();

        // separate candidate lines according to their slope
        ArrayList<Line> pos_lines = new ArrayList<>();
        ArrayList<Line> neg_lines = new ArrayList<>();
        for (Line line : candidate_lines) {
            if (line.slope > 0) pos_lines.add(line);
            if (line.slope < 0) neg_lines.add(line);
        }
        System.out.println(pos_lines.size());
        System.out.println(neg_lines.size());
        System.out.println("-------------------------------");
        // interpolate biases and slopes to compute equation of line that approximates left lane
        // median is employed to filter outliers
        ArrayList<Integer> t_bias = new ArrayList<>();
        ArrayList<Double> t_slopes = new ArrayList<>();
        for (Line line : neg_lines) {
            t_bias.add((int) line.bias);
            t_slopes.add(line.slope);
            System.out.println("bias: " + line.bias + " slope: " + line.slope);
        }
        //neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
        //neg_slope = np.median([l.slope for l in neg_lines])
        //double[] slopes = t_slopes.stream().mapToDouble(Double::doubleValue).toArray();
        //int[] bias = t_bias.stream().mapToInt(Integer::intValue).toArray();
        int neg_bias = (int) Utils.median(t_bias);
        double neg_slope = Utils.median(t_slopes);
        System.out.println("neg_bias: " + neg_bias + " neg_slope: " + neg_slope);
        int x1 = 0, y1 = neg_bias;
        int x2 = -1 * (int) (Math.round(neg_bias / neg_slope));
        int y2 = 0;
        Line left_lane = new Line(x1, y1, x2, y2);

        // interpolate biases and slopes to compute equation of line that approximates right lane
        // median is employed to filter outliers
        t_bias.clear();
        t_slopes.clear();
        System.out.println(t_slopes.size());
        for (Line line : pos_lines) {
            t_bias.add((int) line.bias);
            t_slopes.add(line.slope);
        }
        //lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
        //lane_right_slope = np.median([l.slope for l in pos_lines])
        //double[] r_slopes = t_slopes.stream().mapToDouble(Double::doubleValue).toArray();
        //int[] r_bias = t_bias.stream().mapToInt(Integer::intValue).toArray();
        int lane_right_bias = (int) Utils.median(t_bias); //manager.create(r_bias).median().getInt(0);
        double lane_right_slope = Utils.median(t_slopes); //.median().getInt(0);
        x1 = 0;
        y1 = lane_right_bias;
        x2 = (int) (Math.round((img_shape[0] - lane_right_bias) / lane_right_slope));
        y2 = (int) img_shape[0];
        Line right_lane = new Line(x1, y1, x2, y2);

        lane_lines.add(left_lane);
        lane_lines.add(right_lane);

        double[] val = lane_lines.get(0).get_coords();
        System.out.println(val[0] + " " + val[1] + " " + val[2] + " " + val[3]);
        val = lane_lines.get(1).get_coords();
        System.out.println(val[0] + " " + val[1] + " " + val[2] + " " + val[3]);
    }

    @Test
    public void testPipeLine() throws InterruptedException {
        // Entry point for lane detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
        // with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.
        LaneDetection Ldt = new LaneDetection();
        boolean solid_lines = true;
        boolean temporal_smoothing = true;
        NDManager manager = NDManager.newBaseManager();
        String img_path = "./data/self_driving/images";
        Mat color_image = imread(img_path + "/solidYellowCurve2.jpg", IMREAD_COLOR);
        cvtColor(color_image, color_image, COLOR_BGR2RGB);

        ArrayList<Mat> frames = new ArrayList<>();
        frames.add(color_image);

        boolean is_videoclip = frames.size() > 0;

        int img_h = frames.get(0).rows(), img_w = frames.get(0).cols();

        System.out.println(is_videoclip + " " + img_h + " " + img_w);

        ArrayList<ArrayList<Line>> lane_lines = new ArrayList<>();
        for(int k = 0; k < frames.size(); k++ ) {
            ArrayList<Line> inferred_lanes = Ldt.get_lane_lines(frames.get(k), solid_lines);
            lane_lines.add(inferred_lanes);
        }

        ArrayList<Line> s_lines = new ArrayList<>();

        if( temporal_smoothing && solid_lines ) {
            s_lines = Ldt.smoothen_over_time(lane_lines);
        } else {
            s_lines = lane_lines.get(0);
        }

        // prepare empty mask on which lines are drawn
        NDArray line_img = manager.zeros(new Shape(img_h, img_w), DataType.FLOAT64);
        Mat laneImg = ndarrayToMat(line_img);
        System.out.println("ch:" + laneImg.channels());

        // draw lanes found
        for(Line  lane : s_lines ) {
            lane.draw(laneImg, null, 0);
        }

        ImageViewer.show(laneImg);
        Thread.sleep(100);

        //keep only region of interest by masking
        List<Point> vs = new ArrayList<>();
        vs.add( new Point(50, img_h));
        vs.add( new Point(450, 310));
        vs.add( new Point(490, 310));
        vs.add( new Point(img_w - 50, img_h));
        MatOfPoint mPoints = new MatOfPoint();
        mPoints.fromList(vs);
        ArrayList<MatOfPoint> vertices = new ArrayList<MatOfPoint>();
        vertices.add(mPoints);

        MaskMats rlt = Ldt.region_of_interest(laneImg, vertices);
        ImageViewer.show(rlt.getMasked_image());
        Thread.sleep(100);

        // make blend on color image
        Mat img_color = frames.get(0);
        if( is_videoclip ) img_color = frames.get(frames.size() - 1);
        double a = 0.8, b=1., lambda=0.;
        Mat img_blend = Ldt.weighted_img(rlt.getMasked_image(), img_color, a, b, lambda);

        cvtColor(img_blend, img_blend, COLOR_RGB2BGR);
        ImageViewer.show(img_blend);
        Thread.sleep(500);
    }
}
