package com.example.sudokuopencv

import android.os.Environment
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.ml. *
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import org.opencv.ml.KNearest


class DigitRecognizer {
    private val images_path = "train-images-idx3-ubyte.idx3"
    private val labels_path = "train-labels-idx1-ubyte.idx1"
    var knn: KNearest? = null
    var svm: SVM? = null
    private var total_images = 0
    private var width = 0
    private var height = 0
    fun ReadMNISTData() {
        val external_storage = Environment.getExternalStorageDirectory()
        val mnist_images_file = File(external_storage, images_path)
        val mnist_labels_file = File(external_storage, labels_path)
        var images_reader: FileInputStream? = null
        var labels_reader: FileInputStream? = null
        try {
            images_reader = FileInputStream(mnist_images_file)
            labels_reader = FileInputStream(mnist_labels_file)
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        }
        var training_images = Mat()
        var training_labels = Mat()
        if (images_reader != null) {
            try {
                val header = ByteArray(16)
                images_reader.read(header, 0, 16)

                // Combining the bytes to form an integer
                val temp = ByteBuffer.wrap(header, 4, 12)
                total_images = temp.int
                width = temp.int
                height = temp.int

                //Total number of pixels in each image
                val px_count = width * height
                training_images = Mat(total_images, px_count, CvType.CV_8U)

                //Read each image and store it in an array
                for (i in 0 until total_images) {
                    val image = ByteArray(px_count)
                    images_reader.read(image, 0, px_count)
                    training_images.put(i, 0, image)
                }
                training_images.convertTo(training_images, CvType.CV_32FC1)
                images_reader.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        if (labels_reader != null) {
            // Read Labels
            val labels_data: ByteArray
            labels_data = ByteArray(total_images)
            try {
                training_labels = Mat(total_images, 1, CvType.CV_8U)
                val temp_labels = Mat(1, total_images, CvType.CV_8U)
                val header = ByteArray(8)
                //read the header
                labels_reader.read(header, 0, 8)
                // read all labels at once
                labels_reader.read(labels_data, 0, total_images)
                temp_labels.put(0, 0, labels_data)

                // take a transpose of the image
                Core.transpose(temp_labels, training_labels)
                training_labels.convertTo(training_labels, CvType.CV_32FC1)
                labels_reader.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
            knn?.train(training_images, 10, training_labels)
        }
    }

    fun FindMatch(test_image: Mat): Int {
        Imgproc.dilate(
            test_image, test_image,
            Imgproc.getStructuringElement(
                Imgproc.CV_SHAPE_CROSS,
                Size(3.0, 3.0)
            )
        )
        // Resize the image
        Imgproc.resize(test_image, test_image, Size(width.toDouble(), height.toDouble()))
        // Convert the image to grayscale
//        Imgproc.cvtColor(test_image, test_image, Imgproc.COLOR_RGB2GRAY);
        // Adaptive Threshold
        Imgproc.adaptiveThreshold(
            test_image, test_image, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY_INV, 15, 2.0
        )
        val test = Mat(
            1, test_image.rows() *
                    test_image.cols(), CvType.CV_32FC1
        )
        var count = 0
        for (i in 0 until test_image.rows()) {
            for (j in 0 until test_image.cols()) {
                test.put(0, count, test_image[i, j][0])
                count++
            }
        }
        val results = Mat(1, 1, CvType.CV_8U)
        knn?.findNearest(test, 10, results, Mat(), Mat())
        Log.i("Result:", "" + results[0, 0][0])
        return results[0, 0][0].toInt()
    }
}