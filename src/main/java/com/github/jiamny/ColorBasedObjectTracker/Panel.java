package com.github.jiamny.ColorBasedObjectTracker;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.swing.JPanel;
import org.opencv.core.Mat;

//Kameradan al�nan ve i�lenen g�r�nt�ler jpanel �zerinde g�r�nt�lenecek
class Panel extends JPanel {
	private static final long serialVersionUID = 1L;
	private BufferedImage image;
 
	public Panel() {
		super();
	}
	private BufferedImage getimage() {
		return image;
	}
	public void setimage(BufferedImage newimage) {
		image = newimage;
		return;
	}
 
	public void setimagewithMat(Mat newimage) {
		image = this.ConvertImage(newimage);
		return;
	}
 
	/* Mat nesnesini frame i�erisinde g�stermek i�in BufferedImage tipine
	�eviriyoruz*/
	public BufferedImage ConvertImage(Mat matrix) {
		// Mat nesnesinin s�tun, sat�r ve boyutunu BufferedImage nesnesi i�intutuyoruz
		int cols = matrix.cols();
		int rows = matrix.rows();
		int elemSize = (int) matrix.elemSize();
		byte[] data = new byte[cols * rows * elemSize];
		int type;
		matrix.get(0, 0, data);
		// Mat nesnesinin ka� kanall�, hangi renk uzay�nda oldu�unu tespit ediyoruz.
		switch (matrix.channels()) {
		// Tek kanall� gri renk uzay�na sahip matris
		case 1:
			type = BufferedImage.TYPE_BYTE_GRAY;
			break;
		// �� kanall� BGR renk uzay�na sahip matris
		case 3:
			type = BufferedImage.TYPE_3BYTE_BGR;
	/*
	 * Opencv rgb renk uzay�n� bgr olarak tuttu�u i�in g�r�nt�lemede
	 * d�zg�n bir g�r�nt� elde etmek amac�yla rgb uzay�na �eviriyoruz
	 */
			byte b;
			for (int i = 0; i<data.length; i = i + 3) {
				b = data[i];
				data[i] = data[i + 2];
				data[i + 2] = b;
			}
			break;
		default:
			return null;
		}
		BufferedImage image2 = new BufferedImage(cols, rows, type);
		image2.getRaster().setDataElements(0, 0, cols, rows, data);
		return image2;
	}
 
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
 
		BufferedImage temp = getimage();
 
		if (temp != null)
			g.drawImage(temp, 10, 10, temp.getWidth(), temp.getHeight(), this);
	}
}