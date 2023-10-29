package com.github.jiamny.Social_distance_monitoring;

import ai.djl.engine.Engine;
import org.opencv.core.*;
import org.opencv.dnn.DetectionModel;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static com.github.jiamny.Social_distance_monitoring.DeepSocial.find_zone;
import static org.opencv.core.Core.addWeighted;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.resize;

public class Yolov4DeepSocial {
    static {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
        //System.load("C:\\Program Files\\Opencv4\\java\\x64\\opencv_java454.dll");
    }

    private final static double ReductionFactor = 2.0;
    private static MatOfPoint2f calibration = new MatOfPoint2f(
            new Point(180.0, 162.0),
            new Point(618.0, 0.0),
            new Point(552.0, 540.0),
            new Point(682.0, 464.0)
    );  //{{180,162},{618,0},{552,540},{682,464}};

    //######################## Units are Pixel
    private final static int ViolationDistForIndivisuals = 28;
    private final static int ViolationDistForCouples = 31;
    //####
    private final static int CircleradiusForIndivsual = 14;
    private final static int CircleradiusForCouples = 17;

    //######################## (0:OFF/ 1:ON)
    private final static boolean CorrectionShift = true; // Ignore people in the margins of the video
    private final static int HumanHeightLimit = 200;  // Ignore people with unusual heights
    //########################
    private final static double Transparency = 0.7;

    class CentroidInfo {
        public HashMap<Integer, List<Integer>> centroid = new HashMap<>();
        public Mat image = new Mat();
        public ArrayList<Integer> now_present = new ArrayList<>();
    }

    public CentroidInfo centroid(ArrayList<Rect> detections, ArrayList<Integer> pIds, Mat image, MatOfPoint2f calibration) {
        BirdsEye e = new BirdsEye(image.clone(), calibration);
        HashMap<Integer, List<Integer>> centroid_dict = new HashMap<>();
        CentroidInfo cent = new CentroidInfo();

        ArrayList<Integer> now_present = new ArrayList<>();

        if (detections.size() > 0) {
            for (int i = 0; i < detections.size(); i++) {
                Rect d = detections.get(i);
                int p = pIds.get(i);
                now_present.add(p);
                int w = d.width;
                int h = d.height;
                int x = d.x;
                int y = d.y;
                //System.out.println("h: " + h + " H: " + HumanHeightLimit);
                if(h < HumanHeightLimit) {
                    Mat overley = e.image;
                    Point npt = e.projection_on_bird(new Point((int)(x + w*1.0/2), (int) (y + h)));
                    int bird_x = (int) npt.x, bird_y = (int) npt.y;
                    if (CorrectionShift) {
                        if (DeepSocial.checkupArea(overley, 1.0, 0.25, new Point(x, (int) (y - h * 1.0 / 2)), 'r', false))
                            continue;
                    }

                    e.setImage(overley);
                    Point cpt = e.projection_on_bird(new Point((int)(x + w*1.0/2), (int) (y + h * 1.0 / 2)));
                    int center_bird_x = (int) (cpt.x), center_bird_y = (int) (cpt.y);
                    centroid_dict.put(p, Arrays.asList(bird_x, bird_y, x, (int) (y + h * 1.0 / 2),
                            (int) (x - w * 1.0 / 2), (int) (y), (int) (x + w * 1.0 / 2), (int) (y + h),
                            center_bird_x, center_bird_y));
                }
            }
        }
        // _centroid_dict[p] = centroid_dict[p]
        //return _centroid_dict,centroid_dict, e.image
        cent.centroid = centroid_dict;
        cent.now_present = now_present;
        cent.image = e.image;
        return cent;
    }

    public CentroidInfo detect(Mat img, DetectionModel model, List<String> classes, MatOfPoint2f calibration) {
        MatOfInt classIds = new MatOfInt();
        MatOfFloat scores = new MatOfFloat();
        MatOfRect boxes = new MatOfRect();
        model.detect(img, classIds, scores, boxes, 0.6f, 0.4f);

        ArrayList<Rect> personLst = new ArrayList<>();
        ArrayList<Integer> pIds = new ArrayList<>();
        int pid = 0;
        for (int i = 0; i < classIds.rows(); i++) {
            int classId = (int) classIds.get(i, 0)[0];
            String name = classes.get(classId);

            if (name.equalsIgnoreCase("person")) {
                Rect box = new Rect(boxes.get(i, 0));
                double score = scores.get(i, 0)[0];
                personLst.add(box);
                pid++;
                if (!pIds.contains(Integer.valueOf(pid)))
                    pIds.add(pid);
                /*
                Imgproc.rectangle(img, box, new Scalar(255, 0, 0), 1);

                double score = scores.get(i, 0)[0];
                String text = String.format("%s: %.2f", name, score);
                addTextToBox(img, text, new Scalar(255, 255, 255), Imgproc.FONT_HERSHEY_SIMPLEX, new Scalar(255, 0, 0),
                        new Point(box.x, box.y - 1), 0.3, 1);
                 */
            }
        }
        return centroid(personLst, pIds, img, calibration);
    }

    public static ArrayList<Mat> Apply_ellipticBound(HashMap<Integer, List<Integer>> centroid_dict, Mat img, MatOfPoint2f calibration,
                                              ArrayList<Integer> red, ArrayList<Integer> green, ArrayList<Integer> yellow, ArrayList<Integer> final_redZone,
                                    ArrayList<Integer> coupleZone, ArrayList<Integer> couples, int Single_radius, int Couples_radius) {
        Scalar RedColor = new Scalar(0, 0, 255);
        Scalar GreenColor = new Scalar(0, 255, 0);
        Scalar YellowColor = new Scalar(0, 220, 255);
        Scalar BirdBorderColor = new Scalar(255, 255, 255);
        Scalar BorderColor = new Scalar(220, 220, 220);
        double Transparency = 0.55;
        BirdsEye e = new BirdsEye(img, calibration);
        Mat overlay = e.img2bird();

        ArrayList<Mat> rlt = new ArrayList<>();

        for(Integer i : centroid_dict.keySet() ) {
            int idx = i.intValue();
            int[] box = centroid_dict.get(Integer.valueOf(idx)).stream().mapToInt(j -> j).toArray();

            Point center_bird = new Point(box[0], box[1]);
            if (green.contains(Integer.valueOf(idx))) {
                Imgproc.circle(overlay, center_bird, Single_radius, GreenColor, -1);
                Imgproc.circle(overlay, center_bird, Single_radius, BirdBorderColor, 1);
            }

            if (coupleZone != null) {
                if (red.contains(Integer.valueOf(idx)) && !coupleZone.contains(Integer.valueOf(idx))) {
                    Imgproc.circle(overlay, center_bird, Single_radius, RedColor, -1);
                    Imgproc.circle(overlay, center_bird, Single_radius, BirdBorderColor, 1);
                }
            } else {
                if (red.contains(Integer.valueOf(idx))) {
                    Imgproc.circle(overlay, center_bird, Single_radius, RedColor, -1);
                    Imgproc.circle(overlay, center_bird, Single_radius, BirdBorderColor, 1);
                }
            }
        }

        if( couples != null ) {
                /*
                for p1, p2 in couples {
                    x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
                    centerGroup_bird = e.projection_on_bird((x, ymax))
                    if p1 in yellow:
                        if p2 in yellow:
                            cv2.circle(overlay, centerGroup_bird, Couples_radius, YellowColor, -1)

                    if p1 in final_redZone and p2 in final_redZone:
                        cv2.circle(overlay, centerGroup_bird, Couples_radius, RedColor, -1)
                }

                 */
        }
        e.setBird(overlay);
        Mat dst = new Mat();
        addWeighted(e.original, Transparency,  e.bird2img(), 1 - Transparency, 0.0, dst);
        e.setImage(dst);
        overlay = e.image;

        for(Integer i : centroid_dict.keySet() ) {
            int idx = i.intValue();
            int[] box = centroid_dict.get(Integer.valueOf(idx)).stream().mapToInt(j -> j).toArray();

            Point birdseye_origin = new Point(box[0], box[1]);
            ArrayList<Point> circle_points = e.points_projection_on_image(birdseye_origin, Single_radius);

            if( coupleZone != null ) {
                if (coupleZone.contains(Integer.valueOf(idx))) {
                    for (Point p : circle_points) {
                        Imgproc.circle(overlay, new Point((int) (p.x), (int) (p.y)), 1, BorderColor, -1);
                    }
                }
            }
            int ymin = box[5];
            int ymax = box[7];
            Point origin = e.projection_on_image(new Point(box[0], box[1]));
            int w = 2;
            int x = (int)(origin.x);
            Point top_left = new Point(x - w, ymin);
            Point botton_right = new Point(x + w, ymax);
            if( green.contains(Integer.valueOf(idx)) ) {
                Imgproc.rectangle(overlay, top_left, botton_right, GreenColor, -1);
                Imgproc.rectangle(overlay, top_left, botton_right, BorderColor, 1);
            }
            if( coupleZone != null ) {
                if(red.contains(Integer.valueOf(idx)) && ! coupleZone.contains(Integer.valueOf(idx))) {
                    Imgproc.rectangle(overlay, top_left, botton_right, RedColor, -1);
                    Imgproc.rectangle(overlay, top_left, botton_right, BorderColor, 1);
                }
            } else {
                if(red.contains(Integer.valueOf(idx)) ) {
                    Imgproc.rectangle(overlay, top_left, botton_right, RedColor, -1);
                    Imgproc.rectangle(overlay, top_left, botton_right, BorderColor, 1);
                }
            }
        }
        if( couples != null ) {
            /*
            for p1, p2 in couples {
                x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
                birdseye_origin = e.projection_on_bird((x, ymax))
                circle_points = e.points_projection_on_image(birdseye_origin, Couples_radius)
                for x, y in circle_points:
                    cv2.circle(overlay, (int(x), int(y)),1, BorderColor, -1)

                Point origin = e.projection_on_image(birdseye_origin);
                w = 3
                x = origin[0]
                top_left = (x - w, ymin)
                botton_right = (x + w, ymax)
                if p1 in yellow:
                    if p2 in yellow:
                        cv2.rectangle(overlay, top_left, botton_right, YellowColor, -1)
                        cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
                if p1 in final_redZone and p2 in final_redZone:
                    cv2.rectangle(overlay, top_left, botton_right, RedColor, -1)
                    cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
            }
             */
        }
        e.setImage(overlay);
        rlt.add(e.image);
        rlt.add(e.bird);
        return rlt;
    }


    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        // 确定Deep Java Library (DJL)中的可用GPU内存
        //MemoryUsage mem = CudaUtils.getGpuMemory(device);
        //mem.getMax();
        System.out.println("Engine: " + Engine.getInstance().getEngineName());

        //######################## Frame number
        int StartFrom  = 0;
        int EndAt      = 500;            // -1 for the end of the video

        Yolov4DeepSocial yolo4ds = new Yolov4DeepSocial();
        int width  = 608;                // 416
        int height = 608;

        try {
            List<String> classes = Files.readAllLines(Paths.get("./data/coco.names"));

            Net net = Dnn.readNetFromDarknet("/media/hhj/localssd/DL_data/cfgs/yolov4.cfg",
                                             "/media/hhj/localssd/DL_data/weights/yolo4/yolov4.weights");

            DetectionModel model = new DetectionModel(net);

            VideoCapture cap = new VideoCapture("/media/hhj/localssd/DL_data/videos/OxfordTownCentreDataset.mp4");
            int frame_width = (int)(cap.get(3));
            int frame_height = (int)(cap.get(4));
            width = (int)(frame_width/ReductionFactor);
            height = (int)(frame_height/ReductionFactor);

            model.setInputParams(1 / 255.0, new Size(width, height), new Scalar(0), true);

            if (!cap.isOpened()) {
                System.err.println("Error opening video file");
                cap.release();
                System.exit(-1);
            } else {

                Mat frame = new Mat();          // output mat
                boolean flag = cap.read(frame); // read current frame
                int num_frame = 1;
                while (flag) {
                    //frame_resized = cv2.resize(frame_read,(width, height), interpolation=cv2.INTER_LINEAR)
                    //image = frame_resized
                    resize(frame, frame, new Size(width, height), INTER_LINEAR);
                    System.out.println("r: " + frame.rows() + ", c: " + frame.cols() + " num_frame:" + num_frame);
                    BirdsEye e = new BirdsEye(frame, calibration);
                    CentroidInfo centInfo = yolo4ds.detect(frame, model, classes, calibration);
                    
                    ArrayList<ArrayList<Integer>> fds = find_zone(centInfo.centroid, ViolationDistForIndivisuals);
                    ArrayList<Integer> redZone = fds.get(0);
                    ArrayList<Integer> greenZone = fds.get(1);

                    ArrayList<Integer> redGroups = redZone;
                    ArrayList<Integer> final_redZone = redZone;
                    ArrayList<Mat> rlt = Apply_ellipticBound(centInfo.centroid, frame, calibration, redZone, greenZone,
                            null, final_redZone, null, null, CircleradiusForIndivsual, CircleradiusForCouples);
                    //SDimageVid.write(SDimage)
                    Mat SDimage = rlt.get(0);
                    Mat birdSDimage = rlt.get(1);

                    HighGui.imshow("Detected", SDimage);
                    HighGui.waitKey(1);
                    num_frame++;
                    if( num_frame <= StartFrom )
                        continue;
                    if( num_frame != -1 )
                        if( num_frame > EndAt )
                            break;
                    flag = cap.read(frame);
                }
                HighGui.destroyAllWindows();
                cap.release();
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        System.exit(0);
    }
}
