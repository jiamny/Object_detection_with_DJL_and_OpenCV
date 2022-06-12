import com.github.jiamny.Utils.ImageViewer;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;


import static com.github.jiamny.Utils.ImageHelper.bufferedImage2Mat;

public class ReadPPMimageFileTest {
    @Test
    public void testReadPpmFile() {
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");

        try {
            System.out.println(Arrays.toString(ImageIO.getReaderFormatNames()));

            BufferedImage in_image = ImageIO.read(new File("./data/GTSRB/Final_Test/Images/00000.ppm"));

            ImageViewer.show(bufferedImage2Mat(in_image));
            Thread.sleep(1500);
        }catch(Exception e) {
            e.printStackTrace();
        }
    }
}
