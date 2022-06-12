package com.github.jiamny.Social_distance_monitoring;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import static com.github.jiamny.Utils.ImageHelper.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

class IntrAndExtrMatrix {
    private NDArray intrinsic_matrix = null;
    private NDArray extrinsic_matrix = null;

    public void setExtrinsic_matrix(NDArray extrinsic_matrix) {
        this.extrinsic_matrix = extrinsic_matrix;
    }

    public void setIntrinsic_matrix(NDArray intrinsic_matrix) {
        this.intrinsic_matrix = intrinsic_matrix;
    }

    public NDArray getIntrinsic_matrix() {
        return intrinsic_matrix;
    }

    public NDArray getExtrinsic_matrix() {
        return extrinsic_matrix;
    }
}

public class Oxford_town_calibrate {

    private static int DEFAULT_MULTIPLIER = 10;

    public static IntrAndExtrMatrix projection_matrices_oxford_town() {
        IntrAndExtrMatrix inexMatrix = new IntrAndExtrMatrix();
        try {
            NDManager manager = NDManager.newBaseManager();
            // intrinsic matrix
            double F_X = 2696.35888671875000000000, F_Y = 2696.35888671875000000000,
                    C_X = 959.50000000000000000000, C_Y = 539.50000000000000000000;
            double[][] intrinsic_matrix = {
                    {F_X, 0, C_X},
                    {0, F_Y, C_Y},
                    {0, 0, 1}};
            NDArray nd = manager.create(intrinsic_matrix);
            //System.out.println(nd);
            inexMatrix.setIntrinsic_matrix(nd);
            //extrinsic matrix
            double[][] r_matrix = {
                    {0.46291535, -0.88608972, -0.02354562},
                    {-0.3140051, -0.13908777, -0.93917804},
                    {0.8289211, 0.44215337, -0.34262255}};

            NDArray rotation_matrix = manager.create(r_matrix);
            double[][] trans = {{-0.05988363921642303467}, {3.83331298828125000000}, {12.39112186431884765625}};
            NDArray translation = manager.create(trans);
            NDArray extrinsic_matrix = rotation_matrix.concat(translation, 1);

            inexMatrix.setExtrinsic_matrix(extrinsic_matrix);
        } catch (Exception e) {
            e.printStackTrace();
        }
        //System.out.println(inexMatrix.getExtrinsic_matrix());
        return inexMatrix;
    }

    public static NDArray project_w2c(double[] p, NDArray in_mat, NDArray ex_mat, boolean distortion) {
        NDManager manager = NDManager.newBaseManager();
        //double [] a = p.stream().mapToDouble(Double::doubleValue).toArray();;
        // extrinsic
        NDArray P = manager.create(p).reshape(4, 1);
        NDArray p_temp = ex_mat.matMul(P);
        NDIndex idx = new NDIndex();

        // distortion
        if (distortion) {
            double K1 = -0.60150605440139770508,
                    K2 = 4.70203733444213867188,
                    P1 = -0.00047452122089453042,
                    P2 = -0.00782289821654558182;
            double x_p = p_temp.getDouble(0, 0);    // [0][0]
            double y_p = p_temp.getDouble(1, 0);    // [1][0]
            double r_sq = x_p * x_p + y_p * y_p;
            double xpp = x_p * (1 + K1 * r_sq + K2 * (r_sq * r_sq)) + 2 * P1 * x_p * y_p + P2 * (r_sq + 2 * (x_p * x_p));
            double ypp = y_p * (1 + K1 * r_sq + K2 * (r_sq * r_sq)) + 2 * P2 * x_p * y_p + P1 * (r_sq + 2 * (y_p * y_p));
            p_temp.set(idx.addIndices(0, 0), xpp); //[0][0] = xpp;
            p_temp.set(idx.addIndices(1, 0), ypp); //[1][0] = ypp
        }
        // intrinsic
        P = in_mat.matMul(p_temp);  // @ p_temp
        P = P.div(P.get(2));        // / p[2]
        return P;
    }


    public static void convert_background(int multiplier) {
        try {
            NDManager manager = NDManager.newBaseManager();
            Mat image = Imgcodecs.imread("output/oxford_town_background.png");
            NDArray img = mat2DjlImage(image).toNDArray(manager);
            System.out.println(image.height() + " " + image.width() + " = " + image.rows() + " " + image.cols());
            long HEIGHT = img.getShape().getShape()[0];
            long WIDTH = img.getShape().getShape()[1];
            int DEPTH = image.depth();
            int h_cal = 50 * multiplier;  //10 pixels = 1 meter
            int w_cal = 50 * multiplier;
            NDArray img_cal = manager.zeros(new Shape(h_cal, w_cal, 3));
            IntrAndExtrMatrix inex_mat = projection_matrices_oxford_town();
            NDArray in_mat = inex_mat.getIntrinsic_matrix();
            NDArray ex_mat = inex_mat.getExtrinsic_matrix();

            System.out.println(img_cal.getShape());
            System.out.println(img.getShape());

            for (int i = 0; i < h_cal; i++) {
                for (int j = 0; j < w_cal; j++) {
                    double[] p = {i * 1.0 / multiplier, j * 1.0 / multiplier, 0.0, 1.0};
                    NDArray pp = project_w2c(p, in_mat, ex_mat, false);
                    long x = (long) pp.getDouble(0, 0), y = (long) pp.getDouble(1, 0);

                    if ((0 <= y) && (y < HEIGHT)) {
                        if ((0 <= x) && (x < WIDTH)) {
                            String sid = "" + i + ", " + j + ", :";
                            String gid = "" + y + ", " + x + ", :";
                            img_cal.set(new NDIndex(sid), img.get(new NDIndex(gid))); // [i, j, :] = img[y, x, :]
                        }
                    }
                }
            }

            img_cal = img_cal.toType(DataType.UINT8, false);

            Mat mimg = ndarrayToMat(img_cal);
            if (mimg != null) imwrite("output/oxford_town_background_calibrated_mat.png", mimg);
            /*
            BufferedImageFactory bfm = new BufferedImageFactory();
            Image bimg = bfm.fromNDArray(img_cal);
            Path outputDir = Paths.get("output");
            Path imagePath = outputDir.resolve("oxford_town_background_calibrated.png");
            bimg.save(Files.newOutputStream(imagePath), "png");
             */
            System.out.println("Calibrated background saved.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
        Device device = Engine.getInstance().defaultDevice();
        if (Engine.getInstance().getGpuCount() > 0)
            device = Engine.getInstance().getDevices(1)[0];

        System.out.println(device);
        convert_background(DEFAULT_MULTIPLIER);

    }
}
