package com.github.jiamny.Utils;

import org.opencv.core.Mat;

public class MaskMats {
    private Mat masked_image = new Mat(), mask = new Mat();

    public void setMask(Mat mask) {
        this.mask = mask;
    }

    public void setMasked_image(Mat masked_image) {
        this.masked_image = masked_image;
    }

    public Mat getMasked_image() {
        return masked_image;
    }

    public Mat getMask() {
        return this.mask;
    }

}
