package com.github.jiamny.Self_driving.P1_traffic_lane_finding;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.github.jiamny.Utils.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.util.*;
import java.util.List;

import static com.github.jiamny.Utils.ImageHelper.mat2DjlImage;
import static com.github.jiamny.Utils.ImageHelper.ndarrayToMat;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.*;

public class LaneDetection {
    static {
        // load the OpenCV native library
        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
    }

    public MaskMats region_of_interest(Mat img, ArrayList<MatOfPoint> vertices) {
        MaskMats masknds = new MaskMats();
        NDManager manager = NDManager.newBaseManager();
        // Applies an image mask.
        // Only keeps the region of the image defined by the polygon
        // formed from `vertices`. The rest of the image is set to black.

        //defining a blank mask to start with
        NDArray ndmask = mat2DjlImage(img).toNDArray(manager).zerosLike(); //np.zeros_like(img)
        Mat mask = ImageHelper.ndarrayToMat(ndmask);

        // defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        Scalar ignore_mask_color = null;
        if (img.channels() > 2) {
            int channel_count = img.channels();  //i.e. 3 or 4 depending on your image
            if (img.channels() == 3) ignore_mask_color = new Scalar(255, 255, 255);
            if (img.channels() == 4) ignore_mask_color = new Scalar(255, 255, 255, 255);
        } else {
            ignore_mask_color = new Scalar(255, 255, 255);
        }

        //filling pixels inside the polygon defined by "vertices" with the fill color
        Imgproc.fillPoly(mask, vertices, ignore_mask_color);
        cvtColor(mask, mask, COLOR_RGB2GRAY);

        //returning the image only where mask pixels are nonzero
        Mat masked_image = new Mat();
        Core.bitwise_and(img, mask, masked_image);

        masknds.setMasked_image(masked_image);
        masknds.setMask(mask);
        return masknds;
    }

    public Mat hough_lines_detection(Mat img, double rho, double theta, int threshold, double min_line_len, double max_line_gap) {
        //`img` should be the output of a Canny transform.
        Mat lines = new Mat();
        Imgproc.HoughLinesP(img, lines, rho, theta, threshold, min_line_len, max_line_gap);

        return lines;
    }

    public Mat weighted_img(Mat img, Mat initial_img, double a, double b, double lambda) { //α=0.8, β=1., λ=0.
        // Returns resulting blend image computed as follows:
        // initial_img * α + img * β + λ
        NDManager manager = NDManager.newBaseManager();
        NDArray ndimg = mat2DjlImage(img).toNDArray(manager); //.zerosLike();
        ndimg = ndimg.toType(DataType.UINT8, false);       //np.uint8(img)
        NDArray kp = ndimg.duplicate();
        if(ndimg.getShape().getShape().length == 2) {
            ndimg.stack(kp.zerosLike());
            ndimg.stack(kp.zerosLike());
        }

        img = ImageHelper.ndarrayToMat(ndimg);
        //cvtColor(img, img, COLOR_RGB2BGR);
        Mat mask = new Mat();
        // ----------------------------------------------------------------
        // check white region and change color to red
        // ----------------------------------------------------------------
        Core.inRange(img, new Scalar(255,255,255), new Scalar(255,255,255), mask);
        img.setTo(new Scalar(255,0,0), mask);

        Mat dstImg = new Mat();
        Core.addWeighted(initial_img, a, img, b, lambda, dstImg);
        return dstImg;
    }

    public ArrayList<Line> compute_lane_from_candidates(ArrayList<Line> line_candidates, long[] img_shape) {

        NDManager manager = NDManager.newBaseManager();
        // Compute lines that approximate the position of both road lanes.
        // :param line_candidates: lines from hough transform
        // :param img_shape: shape of image to which hough transform was applied
        // :return: lines that approximate left and right lane position
        ArrayList<Line> result = new ArrayList<>();

        // separate candidate lines according to their slope
        ArrayList<Line> pos_lines = new ArrayList<>();
        ArrayList<Line> neg_lines = new ArrayList<>();
        for (Line line : line_candidates) {
            if (line.slope > 0) {
                pos_lines.add(line);
            } else {
                neg_lines.add(line);
            }
        }
        //System.out.println("pos_lines: " + pos_lines.size() + "; neg_lines: " + neg_lines.size());

        // interpolate biases and slopes to compute equation of line that approximates left lane
        // median is employed to filter outliers
        Line left_lane = new Line();
        ArrayList<Integer> t_bias = new ArrayList<>();
        ArrayList<Double> t_slopes = new ArrayList<>();

        if( neg_lines.size() > 0 ) {
            for (Line line : neg_lines) {
                t_bias.add((int) line.bias);
                t_slopes.add(line.slope);
            }
            //double[] slopes = t_slopes.stream().mapToDouble(Double::doubleValue).toArray();
            //int[] bias = t_bias.stream().mapToInt(Integer::intValue).toArray();
            int neg_bias = (int) Utils.median(t_bias);
            double neg_slope = Utils.median(t_slopes);

            int x1 = 0, y1 = neg_bias;
            int x2 = -1 * (int) (Math.round(neg_bias / neg_slope));
            int y2 = 0;
            left_lane = new Line(x1, y1, x2, y2);
        }

        // interpolate biases and slopes to compute equation of line that approximates right lane
        // median is employed to filter outliers
        t_bias.clear();
        t_slopes.clear();
        Line right_lane = new Line();
        if( pos_lines.size() > 0 ) {
            for (Line line : pos_lines) {
                t_bias.add((int) line.bias);
                t_slopes.add(line.slope);
            }

            int lane_right_bias = (int) Utils.median(t_bias); //manager.create(r_bias).median().getInt(0);
            double lane_right_slope = Utils.median(t_slopes); //.median().getInt(0);
            int x1 = 0, y1 = lane_right_bias;
            int x2 = (int) (Math.round((img_shape[0] - lane_right_bias) / lane_right_slope));
            int y2 = (int) img_shape[0];
            right_lane = new Line(x1, y1, x2, y2);
        }

        result.add(left_lane);
        result.add(right_lane);
        return result;
    }

    public ArrayList<Line> get_lane_lines(Mat color_image, boolean solid_lines) {
        // This function take as input a color road frame and tries to infer the lane lines in the image.
        // :param color_image: input frame
        // :param solid_lines: if True, only selected lane lines are returned. If False, all candidate lines are returned.
        // :return: list of (candidate) lane lines.

        NDManager manager = NDManager.newBaseManager();
        int resize_h = 540, resize_w = 960;
        // resize to 960 x 540
        resize(color_image, color_image, new Size(resize_w, resize_h));

        // convert to grayscale
        Mat img_gray = new Mat();
        cvtColor(color_image, img_gray, COLOR_RGB2GRAY);

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
        Mat lines = hough_lines_detection(img_edge, rho, theta, threshold, min_line_len, max_line_gap);
        //Imgproc.HoughLinesP(img_edge, lines, rho, theta, threshold, min_line_len, max_line_gap);

        ArrayList<Line> d_lines = new ArrayList<>();

        for (int i = 0; i < lines.rows(); i++) {
            double[] val = lines.get(i, 0);
            d_lines.add(new Line(val[0], val[1], val[2], val[3]));
        }

        ArrayList<Line> lane_lines = new ArrayList<>();
        // if 'solid_lines' infer the two lane lines
        if( solid_lines ) {
            ArrayList<Line> candidate_lines = new ArrayList<>();
            for( Line line : d_lines )
                // consider only lines with slope between 30 and 60 degrees
                if( (0.5 <= Math.abs(line.slope)) && (Math.abs(line.slope) <= 2) ) {
                    //System.out.println(line.get_coords()[0] + " " + line.get_coords()[1] + " " + line.get_coords()[2] + " " + line.get_coords()[3]);
                    candidate_lines.add(line);
                }
            // System.out.println(candidate_lines.size());
            // interpolate lines candidates to find both lanes
            lane_lines = compute_lane_from_candidates(candidate_lines, mat2DjlImage(img_gray).toNDArray(manager).getShape().getShape());
        } else {
            // if not solid_lines, just return the hough transform output
            lane_lines = d_lines;
        }
        return lane_lines;
    }

    public ArrayList<Line> smoothen_over_time(ArrayList<ArrayList<Line>> lane_lines) {
        // Smooth the lane line inference over a window of frames and returns the average lines.
        NDManager manager = NDManager.newBaseManager();
        ArrayList<Line> avg_lines = new ArrayList<>();

        NDArray avg_line_lt = manager.zeros(new Shape(lane_lines.size(), 4)).toType(DataType.FLOAT64, false);
        NDArray avg_line_rt = manager.zeros(new Shape(lane_lines.size(), 4)).toType(DataType.FLOAT64, false);

        for( int  t = 0; t < lane_lines.size(); t++ ) {
            String sid = "" + t + ", :";
            String gid = ":";
            NDArray ltm = avg_line_lt.get(new NDIndex(sid)).duplicate();
            ltm = ltm.add(manager.create(lane_lines.get(t).get(0).get_coords())); // left lane
            //System.out.println("ltm: " + ltm.getShape());
            avg_line_lt.set(new NDIndex(sid), ltm.get(new NDIndex(gid)));

            NDArray rtm = avg_line_rt.get(new NDIndex(sid)).duplicate();
            rtm = rtm.add(manager.create(lane_lines.get(t).get(1).get_coords())); // right lane
            //System.out.println("rtm: " + rtm.getShape());
            avg_line_rt.set(new NDIndex(sid), rtm.get(new NDIndex(gid))) ;
        }
        double[] ltval = avg_line_lt.mean(new int[]{0}).toDoubleArray();
        double[] rtval = avg_line_rt.mean(new int[]{0}).toDoubleArray();

        avg_lines.add(new Line(ltval[0], ltval[1], ltval[2], ltval[3]));
        avg_lines.add(new Line(rtval[0], rtval[1], rtval[2], rtval[3]));
        return avg_lines;
    }

    public Mat color_frame_pipeline(ArrayList<Mat> frames, boolean solid_lines, boolean temporal_smoothing) {
        // Entry point for lane detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
        // with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.

        NDManager manager = NDManager.newBaseManager();
        boolean is_videoclip = false;
        if( frames.size() > 0 ) is_videoclip = true;

        int img_h = frames.get(0).rows(), img_w = frames.get(0).cols();

        ArrayList<ArrayList<Line>> lane_lines = new ArrayList<>();
        for(int k = 0; k < frames.size(); k++ ) {
            ArrayList<Line> inferred_lanes = get_lane_lines(frames.get(k), solid_lines);
            lane_lines.add(inferred_lanes);
        }

        ArrayList<Line> s_lines = new ArrayList<>();

        if( temporal_smoothing && solid_lines ) {
            s_lines = smoothen_over_time(lane_lines);
        } else {
            s_lines = lane_lines.get(0);
        }

        // prepare empty mask on which lines are drawn
        NDArray line_img = manager.zeros(new Shape(img_h, img_w), DataType.FLOAT64);
        Mat laneImg = ndarrayToMat(line_img);

        // draw lanes found
        for(Line  lane : s_lines ) {
            lane.draw(laneImg, null, 0);
        }

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

        MaskMats rlt = region_of_interest(laneImg, vertices);

        // make blend on color image
        Mat img_color = frames.get(0);
        if( is_videoclip ) img_color = frames.get(frames.size() - 1);
        double a = 0.8, b=1., lambda=0.;
        Mat img_blend = weighted_img(rlt.getMasked_image(), img_color, a, b, lambda);

        return img_blend;
    }

    public static void main(String[] args) {
        String current_dir = System.getProperty("user.dir");
        System.out.println(current_dir);

        LaneDetection Ldt = new LaneDetection();
        String img_path = "./data/self_driving/images";
        String out_path = "./output/images";

        boolean solid_lines = true;
        boolean temporal_smoothing = true;
        ArrayList<Mat> frames = new ArrayList<>();

        try {
            // test on images
            File[] files = new File(img_path).listFiles();
            if (files.length > 0) {

                for (File file : files) {
                    frames.clear();
                    System.out.println(file.getName());
                    //System.out.println(files[0]);
                    Mat color_image = imread(img_path + "/" + file.getName(), IMREAD_COLOR);
                    cvtColor(color_image, color_image, COLOR_BGR2RGB);
                    frames.add(color_image);
                    Mat img_blend = Ldt.color_frame_pipeline(frames, solid_lines, temporal_smoothing);

                    cvtColor(img_blend, img_blend, COLOR_RGB2BGR);
                    HighGui.imshow(file.getName(), img_blend);
                    HighGui.waitKey(50);
                    Imgcodecs.imwrite(out_path + "/" + file.getName(), img_blend);
                }
                HighGui.destroyAllWindows();
            }

            // test on videos
            int resize_h = 540, resize_w = 960;
            String f = "./data/self_driving/videos/solidWhiteRight.mp4";
            String of = "./output/videos/solidWhiteRight.mp4";
            boolean useImshow = true;
            String tlt = "Traffic_lane_finding";
            VideoPlay vp =new VideoPlay();
            Mat color_image = new Mat();
            VideoCapture cap = new VideoCapture();
            cap.open(f);
            //VideoWriter.fourcc(*'DIVX')
            VideoWriter out = new VideoWriter(of, VideoWriter.fourcc('m','p','4','v'), 20.0, new Size(resize_w, resize_w), true);

            if (cap.isOpened()) {
                while (true) {
                    cap.read(color_image);
                    if( ! color_image.empty() ) {
                        frames.clear();
                        cvtColor(color_image, color_image, COLOR_BGR2RGB);
                        frames.add(color_image);

                        Mat img_blend = Ldt.color_frame_pipeline(frames, solid_lines, temporal_smoothing);
                        cvtColor(img_blend, img_blend, COLOR_RGB2BGR);
                        int k = vp.displayImage(img_blend, tlt, useImshow);
                        if( k > 0 ) break;
                    } else {
                        break;
                    }
                }
            }

            if( useImshow ) {
                HighGui.destroyAllWindows();
            }
            cap.release();
            out.release();
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}
