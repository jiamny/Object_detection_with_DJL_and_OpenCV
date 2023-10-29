package com.github.jiamny.ColorBasedObjectTracker;

import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class Detector {

	public static void main(String arg[]) {

		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.load("/usr/local/share/java/opencv4/libopencv_java480.so");

		// Anl�kolarakyakalanankamerag�r�nt�lerinig�sterece�imiz frame ve panel
		JFrame cameraFrame = new JFrame("Anl�k kamera g�r�nt�s�");
		cameraFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		cameraFrame.setSize(640, 480);
		cameraFrame.setBounds(0, 0, cameraFrame.getWidth(), cameraFrame.getHeight());
		Panel panelCamera = new Panel();
		cameraFrame.setContentPane(panelCamera);
		cameraFrame.setVisible(true);

		// ��lenecekg�r�nt�n�n threshold uyguland�ktan sonraki halini
		// g�sterece�imiz frame ve panel
		JFrame thresholdFrame = new JFrame("Threshold g�r�nt�");
		thresholdFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		thresholdFrame.setSize(640, 480);
		thresholdFrame.setBounds(0, 0, cameraFrame.getWidth(), cameraFrame.getHeight());
		Panel panelThreshold = new Panel();
		thresholdFrame.setContentPane(panelThreshold);
		thresholdFrame.setVisible(true);

		// Video ak���i�in, 0 ilevarsay�lankameray�ba�lataca��z
		VideoCapture capture = new VideoCapture();
		// Parametreleri at�yoruz
		/*capture.set(3, 1366);
		capture.set(4, 768);
		capture.set(15, -2);*/
		// Saf kamerag�r�nr�s�
		Mat webcam_image = new Mat();
		// Hsv renk uzay�nda g�r�nt�s�
		Mat hsv_image = new Mat();
		// 1. ve 2. threshold
		Mat thresholded = new Mat();
		Mat thresholded2 = new Mat();
		// Kameradan g�r�nt� oku

		String f = "./data/mulballs.mp4";
		capture.open(f);

		capture.read(webcam_image);
		// Kameradan al�nan g�r�nt�leri g�sterecek oldu�umuz frame boyutlar�
		// kameradan okunang �r�nt�ye g�re ayarlan�yor.

		cameraFrame.setSize(webcam_image.width() + 50, webcam_image.height() + 50);
		thresholdFrame.setSize(webcam_image.width() + 50, webcam_image.height() + 50);

		Mat array255 = new Mat(webcam_image.height(), webcam_image.width(), CvType.CV_8UC1);
		array255.setTo(new Scalar(255));
		Mat distance = new Mat(webcam_image.height(), webcam_image.width(), CvType.CV_8UC1);

		List<Mat> lhsv = new ArrayList<Mat>(3);
		Mat circles = new Mat();
		// Renktespitiburadabelirtti�imizminve max de�erlereg�reyap�lacak
		// hsvuzay�ndaverdi�imizrenktonlar�aras�ndaki her renktespitedilecektir.
		// Renkaral�klar�i�inhsvrenktablolar�nag�zatabilirsiniz.

		Scalar minColor = new Scalar(5, 100, 100, 0);
		Scalar maxColor = new Scalar(10, 255, 255, 0);
		// Kamera ayg�t� �al���yor ise
		if (capture.isOpened()) {
			while (true) {
				capture.read(webcam_image);
				// Bir g�r�nt� okunmu� ve bo�de�ilse
				if (!webcam_image.empty()) {
					// Kameradan okunan g�r�nt� hsv renk uzay�na d�n��t�r�l�r
					Imgproc.cvtColor(webcam_image, hsv_image, Imgproc.COLOR_BGR2HSV);
					Core.inRange(hsv_image, minColor, maxColor, thresholded);

					// Erode ve dilate i�lemi uygulan�r yap�sal element �l��leri
					// iki i�lemdede ayn�d�r
					Imgproc.erode(thresholded, thresholded,
							Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
					Imgproc.dilate(thresholded, thresholded,
							Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
					/*
					 * Split metodu
					 * ileg�r�nt��zerindekihistogramlar�par�al�yoruz. Matrisin
					 * her boyutu ayr� ayr� nesnelere atan�yor.
					 */
					Core.split(hsv_image, lhsv);
					Mat S = lhsv.get(1);
					Mat V = lhsv.get(2);
					// dizileraras� element farklar�hesapl�yoruz
					Core.subtract(array255, S, S);
					Core.subtract(array255, V, V);
					S.convertTo(S, CvType.CV_32F);
					V.convertTo(V, CvType.CV_32F);
					// 2 boyutluvekt�rlerimizinb�y�kl���n�hesapl�yoruz
					Core.magnitude(S, V, distance);
					/*
					 * Verilende�erleraras�nda thresholding uyguluyor.
					 * pikselinde�eriverilende�erleraras�ndaise o
					 * piksel,beyazde�ilsesiyahyap�l�yor.
					 */
					Core.inRange(distance, new Scalar(0.0), new Scalar(200.0), thresholded2);
					Core.bitwise_and(thresholded, thresholded2, thresholded);
					// thresholded i�ingaussian blur filtresiuyguluyoruz
					Imgproc.GaussianBlur(thresholded, thresholded, new Size(9, 9), 0, 0);
					// �eklind��hatlar�
					List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
					Imgproc.HoughCircles(thresholded, circles, Imgproc.CV_HOUGH_GRADIENT, 2, thresholded.height() / 8,
							200, 100, 0, 0);
					// thresholding sonras�nesnenin binary
					// hali�zerindeba�l�noktalar�tesi�ediyoruz
					Imgproc.findContours(thresholded, contours, thresholded2, Imgproc.RETR_LIST,
							Imgproc.CHAIN_APPROX_SIMPLE);
					// Nesneninkonumunaa�a��dakirenkile�iziyoruz
					Imgproc.drawContours(webcam_image, contours, -2, new Scalar(10, 0, 0), 4);

					/*
					 * Paneller�zerineg�r�nr�leriatay�p frame'lerin
					 * tekrardan�izilmesinisa�l�yoruz
					 */
					panelCamera.setimagewithMat(webcam_image);
					panelThreshold.setimagewithMat(thresholded);
					cameraFrame.repaint();
					thresholdFrame.repaint();

				} else {
					JOptionPane.showMessageDialog(null, "Kamera ayg�t�na ba�lan�lamad�!");
					break;
				}
			}
		}
	}
}