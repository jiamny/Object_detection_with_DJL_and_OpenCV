package com.github.jiamny.Utils;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

import static org.opencv.highgui.HighGui.*;


public class VideoPlay {
    private JFrame frame;
    private JLabel imageLabel;

    public void initGUI(String frameName) {
        frame = new JFrame(frameName);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);
        imageLabel = new JLabel();
        frame.add(imageLabel);
        frame.setVisible(true);
    }

    public int displayImage(Mat currentImage, String title, boolean useImshow) throws InterruptedException {
        if (useImshow) {
            imshow(title, currentImage);
            return waitKey(50); // waiting millisec
        } else {
            BufferedImage tempImage = (BufferedImage) HighGui.toBufferedImage(currentImage);
            ImageIcon imageIcon = new ImageIcon(tempImage, title);
            imageLabel.setIcon(imageIcon);
            frame.pack(); // this will resize the window to fit the image
            Thread.sleep(50);
            return 0;
        }
    }
}
