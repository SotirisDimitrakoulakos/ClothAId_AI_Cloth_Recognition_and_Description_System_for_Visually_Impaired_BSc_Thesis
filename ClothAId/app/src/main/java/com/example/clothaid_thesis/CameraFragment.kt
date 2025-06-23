@file:SuppressLint("ClickableViewAccessibility")
package com.example.clothaid_thesis

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.media.MediaPlayer
import android.os.Bundle
import android.view.*
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.clothaid_thesis.databinding.FragmentCameraBinding
import com.google.common.util.concurrent.ListenableFuture
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import android.view.MotionEvent
import android.view.GestureDetector


class CameraFragment : Fragment() {

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private var imageCapture: ImageCapture? = null
    private var outputDirectory: File? = null
    private var photoFile: File? = null
    private var mediaPlayer: MediaPlayer? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private lateinit var gestureDetector: GestureDetector
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var permissionGranted = false


    // Handle permission result
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(requireContext(), "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    private fun toggleCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        outputDirectory = requireContext().getExternalFilesDir(null) ?: requireContext().filesDir


        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            permissionGranted = true
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        // Gesture detector
        gestureDetector = GestureDetector(requireContext(), object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                playCamSwitchSound()
                toggleCamera()
                return true
            }

            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                (activity as? MainActivity)?.stopSpeaking()
                playShutterSound()
                takePhoto()
                return true
            }

            override fun onFling(e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float): Boolean {
                (activity as? MainActivity)?.stopSpeaking()
                if (e1 == null) return false
                val deltaX = e2.x - e1.x
                if (abs(deltaX) > SWIPE_THRESHOLD && abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                    if (deltaX < 0) {
                        openMostRecentGalleryImage()
                        return true
                    }
                }
                return false
            }


            override fun onLongPress(e: MotionEvent) {
                val prompt = "Camera is active"
                Toast.makeText(requireContext(), prompt, Toast.LENGTH_SHORT).show()
                (activity as? MainActivity)?.speak(prompt)
            }
        })

        binding.root.apply {
            isClickable = true
            isFocusable = true

            setOnTouchListener { v, event ->
                gestureDetector.onTouchEvent(event)
                if (event.action == MotionEvent.ACTION_UP) {
                    v.performClick()  // This is fine; suppress warning with @SuppressLint
                }
                true
            }

            // To handle accessibility/talkback click too
            setOnClickListener {
                // No-op
            }
        }
    }

    private fun openMostRecentGalleryImage() {
        val recentImages = GalleryNavigator.getRecentImagePaths(requireContext())
        if (recentImages.isNotEmpty()) {
            val mostRecent = recentImages[0]
            requireActivity().supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, ConfirmFragment.newInstance(mostRecent, false))
                .addToBackStack(null)
                .commit()
            playPageFlipSound()
        } else {
            Toast.makeText(requireContext(), "No recent images found", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageCapture = ImageCapture.Builder().build()
            this.imageCapture = imageCapture

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return

        val fileName = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis()) + ".jpg"
        val photoFile = File(outputDirectory, fileName)
        this.photoFile = photoFile

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(requireContext(), "Photo capture failed: ${exc.message}", Toast.LENGTH_LONG).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val photoPath = photoFile.absolutePath
                    requireActivity().runOnUiThread {
                        requireActivity().supportFragmentManager
                            .beginTransaction()
                            .replace(R.id.fragment_container,
                                ConfirmFragment.newInstance(photoPath, true)
                            )
                            .addToBackStack(null)
                            .commit()
                    }
                }
            }
        )
    }

    private fun playShutterSound() {
        mediaPlayer = MediaPlayer.create(requireContext(), R.raw.camera_shutter)
        mediaPlayer?.setOnCompletionListener {
            it.release()
        }
        mediaPlayer?.start()
    }

    private fun playCamSwitchSound() {
        mediaPlayer = MediaPlayer.create(requireContext(), R.raw.camera_switch)
        mediaPlayer?.setOnCompletionListener {
            it.release()
        }
        mediaPlayer?.start()
    }

    private fun playPageFlipSound() {
        val mediaPlayer = MediaPlayer.create(requireContext(), R.raw.pageturn)
        mediaPlayer.setOnCompletionListener { it.release() }
        mediaPlayer.start()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        mediaPlayer?.release()
    }

    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val SWIPE_THRESHOLD = 100
        private const val SWIPE_VELOCITY_THRESHOLD = 100
    }


}
