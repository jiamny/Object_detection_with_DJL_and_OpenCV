package com.github.jiamny.Utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.util.RandomUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class ImageUtils {

    public static BufferedImage showImages(
            BufferedImage[] images, String[] labels, int width, int height) {
        int col = Math.min(1280 / width, images.length);
        int row = (images.length + col - 1) / col;

        int textHeight = 28;
        int w = col * (width + 3);
        int h = row * (height + 3) + textHeight;
        BufferedImage output = new BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = output.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setPaint(Color.LIGHT_GRAY);
        g.fill(new java.awt.Rectangle(0, 0, w + 3, h + 3));
        g.setPaint(Color.BLACK);

        Font font = g.getFont();
        FontMetrics metrics = g.getFontMetrics(font);
        for (int i = 0; i < images.length; ++i) {
            int x = (i % col) * (width + 3) + 3;
            int y = (i / col) * (height + 3) + 3;

            int tx = x + (width - metrics.stringWidth(labels[i])) / 2;
            int ty = y + ((textHeight - metrics.getHeight()) / 2) + metrics.getAscent();
            g.drawString(labels[i], tx, ty);

            BufferedImage img = images[i];
            g.drawImage(img, x, y + textHeight, width, height, null);
        }
        g.dispose();
        return output;
    }

    public static BufferedImage showImages(BufferedImage[] images, int width, int height) {
        int col = Math.min(1280 / width, images.length);
        int row = (images.length + col - 1) / col;

        int w = col * (width + 3);
        int h = row * (height + 3);
        BufferedImage output = new BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = output.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setPaint(Color.LIGHT_GRAY);
        g.fill(new java.awt.Rectangle(0, 0, w + 3, h + 3));
        for (int i = 0; i < images.length; ++i) {
            int x = (i % col) * (width + 3) + 3;
            int y = (i / col) * (height + 3) + 3;

            BufferedImage img = images[i];
            g.drawImage(img, x, y, width, height, null);
        }
        g.dispose();
        return output;
    }

    public static void drawBBoxes(Image img, NDArray boxes, String[] labels) {
        if (labels == null) {
            labels = new String[(int) boxes.size(0)];
            Arrays.fill(labels, "");
        }

        List<String> classNames = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> boundBoxes = new ArrayList<>();
        for (int i = 0; i < boxes.size(0); i++) {
            NDArray box = boxes.get(i);
            Rectangle rect = bboxToRect(box);
            classNames.add(labels[i]);
            prob.add(1.0);
            boundBoxes.add(rect);
        }
        DetectedObjects detectedObjects = new DetectedObjects(classNames, prob, boundBoxes);
        img.drawBoundingBoxes(detectedObjects);
    }

    public static Rectangle bboxToRect(NDArray bbox) {
        float width = bbox.getFloat(2) - bbox.getFloat(0);
        float height = bbox.getFloat(3) - bbox.getFloat(1);
        return new Rectangle(bbox.getFloat(0), bbox.getFloat(1), width, height);
    }

    public static Image bufferedImage2DJLImage(BufferedImage img) {
        return ImageFactory.getInstance().fromImage(img);
    }

    public static void saveImage(BufferedImage img, String name, String path) {
        Image newImage = ImageFactory.getInstance().fromImage(img); // 支持多种图片格式，自动适配
        Path outputDir = Paths.get(path);
        Path imagePath = outputDir.resolve(name);
        // OpenJDK 不能保存 jpg 图片的 alpha channel
        try {
            newImage.save(Files.newOutputStream(imagePath), "png");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveImage(Image img, String name, String path) {
        Path outputDir = Paths.get(path);
        Path imagePath = outputDir.resolve(name);
        // OpenJDK 不能保存 jpg 图片的 alpha channel
        try {
            img.save(Files.newOutputStream(imagePath), "png");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveBoundingBoxImage(
            Image img, DetectedObjects detection, String name, String path) throws IOException {
        // Make image copy with alpha channel because original image was jpg
        img.drawBoundingBoxes(detection);
        Path outputDir = Paths.get(path);
        Files.createDirectories(outputDir);
        Path imagePath = outputDir.resolve(name);
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
    }

    public static void drawImageRect(BufferedImage image, int x, int y, int width, int height) {
        // 将绘制图像转换为Graphics2D
        Graphics2D g = (Graphics2D) image.getGraphics();
        try {
            g.setColor(new Color(246, 96, 0));
            // 声明画笔属性 ：粗 细（单位像素）末端无修饰 折线处呈尖角
            BasicStroke bStroke = new BasicStroke(4, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
            g.setStroke(bStroke);
            g.drawRect(x, y, width, height);

        } finally {
            g.dispose();
        }
    }

    public static void drawImageRect(
            BufferedImage image, int x, int y, int width, int height, Color c) {
        // 将绘制图像转换为Graphics2D
        Graphics2D g = (Graphics2D) image.getGraphics();
        try {
            g.setColor(c);
            // 声明画笔属性 ：粗 细（单位像素）末端无修饰 折线处呈尖角
            BasicStroke bStroke = new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
            g.setStroke(bStroke);
            g.drawRect(x, y, width, height);

        } finally {
            g.dispose();
        }
    }

    public static void drawImageText(BufferedImage image, String text) {
        Graphics graphics = image.getGraphics();
        int fontSize = 100;
        Font font = new Font("楷体", Font.PLAIN, fontSize);
        try {
            graphics.setFont(font);
            graphics.setColor(new Color(246, 96, 0));
            int strWidth = graphics.getFontMetrics().stringWidth(text);
            graphics.drawString(text, fontSize - (strWidth / 2), fontSize + 30);
        } finally {
            graphics.dispose();
        }
    }

    /** 返回外扩人脸 factor = 1, 100%, factor = 0.2, 20% */
    public static Image getSubImage(Image img, BoundingBox box, float factor) {
        Rectangle rect = box.getBounds();
        // 左上角坐标
        int x1 = (int) (rect.getX() * img.getWidth());
        int y1 = (int) (rect.getY() * img.getHeight());
        // 宽度，高度
        int w = (int) (rect.getWidth() * img.getWidth());
        int h = (int) (rect.getHeight() * img.getHeight());
        // 左上角坐标
        int x2 = x1 + w;
        int y2 = y1 + h;

        // 外扩大100%，防止对齐后人脸出现黑边
        int new_x1 = Math.max((int) (x1 + x1 * factor / 2 - x2 * factor / 2), 0);
        int new_x2 = Math.min((int) (x2 + x2 * factor / 2 - x1 * factor / 2), img.getWidth() - 1);
        int new_y1 = Math.max((int) (y1 + y1 * factor / 2 - y2 * factor / 2), 0);
        int new_y2 = Math.min((int) (y2 + y2 * factor / 2 - y1 * factor / 2), img.getHeight() - 1);
        int new_w = new_x2 - new_x1;
        int new_h = new_y2 - new_y1;
        return img.getSubImage(new_x1, new_y1, new_w, new_h);
    }

    public static void drawJoints(Image img, Image subImg, int x, int y, Joints joints) {
        BufferedImage image = (BufferedImage) img.getWrappedImage();
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke((float) stroke));
        int imageWidth = subImg.getWidth();
        int imageHeight = subImg.getHeight();
        Iterator iterator = joints.getJoints().iterator();

        while (iterator.hasNext()) {
            Joints.Joint joint = (Joints.Joint) iterator.next();
            g.setPaint(randomColor().darker());
            int newX = x + (int) (joint.getX() * (double) imageWidth);
            int newY = y + (int) (joint.getY() * (double) imageHeight);
            g.fillOval(newX, newY, 10, 10);
        }

        g.dispose();
    }

    private static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    public static void drawBoundingBoxImage(Image img, DetectedObjects detection) {
        img.drawBoundingBoxes(detection);
    }

    public static int getX(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        // 左上角x坐标
        int x = (int) (rect.getX() * img.getWidth());
        return x;
    }

    public static int getY(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        // 左上角y坐标
        int y = (int) (rect.getY() * img.getHeight());
        return y;
    }
}
