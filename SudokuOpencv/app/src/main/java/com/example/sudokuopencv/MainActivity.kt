package com.example.sudokuopencv

import android.os.Bundle
import android.view.SurfaceView
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        setContentView(R.layout.activity_main)
        mOpenCvCameraView = findViewById<CameraBridgeViewBase>(R.id.view)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)

    }

    override fun onResume() {
        super.onResume()
        OpenCVLoader.initDebug()
        mOpenCvCameraView!!.enableView()
    }

    override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val grayMat = inputFrame.gray()
        val blurMat = Mat()
        Imgproc.GaussianBlur(grayMat, blurMat, Size(5.0, 5.0), 0.0)
        val thresh = Mat()
        Imgproc.adaptiveThreshold(blurMat, thresh, 255.0,1,1,11,2.0)

        val contours: List<MatOfPoint> = ArrayList()
        val hier = Mat()
        Imgproc.findContours(thresh, contours, hier, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        hier.release()

        var biggest = MatOfPoint2f()
        var max_area = 0.0
        for (i in contours) {
            val area = Imgproc.contourArea(i)
            if (area > 100) {
                val m = MatOfPoint2f(*i.toArray())
                val peri = Imgproc.arcLength(m, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(m, approx, 0.02 * peri, true)
                if (area > max_area && approx.total() == 4L) {
                    biggest = approx
                    max_area = area
                }
            }
        }
        val displayMat = inputFrame.rgba()
        val points = biggest.toArray()
        var cropped = Mat()
        if (points.size >= 4) {
            Imgproc.line(
                displayMat,
                Point(points[0].x, points[0].y),
                Point(points[1].x, points[1].y),
                Scalar(255.0, 0.0, 0.0),
                2
            )
            Imgproc.line(
                displayMat,
                Point(points[1].x, points[1].y),
                Point(points[2].x, points[2].y),
                Scalar(255.0, 0.0, 0.0),
                2
            )
            Imgproc.line(
                displayMat,
                Point(points[2].x, points[2].y),
                Point(points[3].x, points[3].y),
                Scalar(255.0, 0.0, 0.0),
                2
            )
            Imgproc.line(
                displayMat,
                Point(points[3].x, points[3].y),
                Point(points[0].x, points[0].y),
                Scalar(255.0, 0.0, 0.0),
                2
            )
            // crop the image
            val R = Rect(
                Point(points[0].x , points[0].y),
                Point(points[2].x, points[2].y)
            )
            if (displayMat.width() > 1 && displayMat.height() > 1) {
                cropped = Mat(displayMat, R)
            }

        }
        return displayMat
    }
}
