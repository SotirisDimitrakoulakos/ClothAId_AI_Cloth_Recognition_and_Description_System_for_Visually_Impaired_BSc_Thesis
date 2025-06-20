package com.example.clothaid_thesis

import android.content.Context
import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.os.Bundle
import android.util.AttributeSet
import android.view.*
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.clothaid_thesis.databinding.FragmentConfirmBinding
import java.io.File
import kotlin.math.abs

class ConfirmFragment : Fragment() {

    private var _binding: FragmentConfirmBinding? = null
    private val binding get() = _binding!!

    private lateinit var imagePaths: List<String>
    private var currentIndex: Int = 0
    private var lastSwipeTime = 0L
    private var touchStartX = 0f
    private var hasSwiped = false

    private var mediaPlayer: MediaPlayer? = null

    private var imagePath: String? = null
    private var isFromCameraCapture = false

    companion object {
        private const val SWIPE_THRESHOLD = 100

        private const val ARG_IMAGE_PATH = "image_path"
        private const val ARG_FROM_CAMERA = "from_camera"
        fun newInstance(imagePath: String, isFromCameraCapture: Boolean = false): ConfirmFragment {
            val fragment = ConfirmFragment()
            val args = Bundle()
            args.putString(ARG_IMAGE_PATH, imagePath)
            args.putBoolean(ARG_FROM_CAMERA, isFromCameraCapture)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            imagePath = it.getString(ARG_IMAGE_PATH)
            isFromCameraCapture = it.getBoolean(ARG_FROM_CAMERA, false)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentConfirmBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if (isFromCameraCapture) {
            // Only show the single captured image, disable swiping
            imagePaths = listOf(imagePath ?: "")
            currentIndex = 0
        } else {
            // Load recent gallery images for swiping
            imagePaths = GalleryNavigator.getRecentImagePaths(requireContext())
            val initialPath = arguments?.getString(ARG_IMAGE_PATH)
            currentIndex = imagePaths.indexOfFirst { it == initialPath }.takeIf { it >= 0 } ?: 0
        }

        showImageAt(currentIndex)

        if (!isFromCameraCapture) {
            // Enable swipe gestures only when not from camera capture
            binding.imageView.apply {
                setOnTouchListener { v, event ->
                    when (event.action) {
                        MotionEvent.ACTION_DOWN -> {
                            touchStartX = event.x
                            hasSwiped = false
                            false
                        }
                        MotionEvent.ACTION_UP -> {
                            v.performClick()
                            val deltaX = event.x - touchStartX
                            val now = System.currentTimeMillis()
                            if (abs(deltaX) > SWIPE_THRESHOLD && now - lastSwipeTime > 300) {
                                lastSwipeTime = now
                                hasSwiped = true
                                if (deltaX > 0) swipeRight() else swipeLeft()
                            } else {
                                confirmImage()
                            }
                            true
                        }
                        else -> false
                    }
                }
                setOnClickListener {
                    // Optional: handle click if needed
                }
            }
        } else {
            // From camera capture: tap to confirm only, no swipe
            binding.imageView.setOnClickListener {
                confirmImage()
            }
        }

        binding.imageView.setOnLongClickListener {
            if (!hasSwiped) {
                requireActivity().supportFragmentManager.popBackStack()
                true
            } else {
                false // ignore long press if just swiped
            }
        }
    }

    private fun swipeLeft() {
        if (currentIndex < imagePaths.size - 1) {
            currentIndex++
            showImageAt(currentIndex)
            playSwipeSound()
        }
    }

    private fun swipeRight() {
        if (currentIndex > 0) {
            currentIndex--
            showImageAt(currentIndex)
            playSwipeSound()
        }
    }

    private fun showImageAt(index: Int) {
        val path = imagePaths.getOrNull(index) ?: return
        val bitmap = BitmapFactory.decodeFile(path)
        binding.imageView.setImageBitmap(bitmap)
    }

    private fun confirmImage() {
        val path = imagePaths.getOrNull(currentIndex)
        if (path != null && File(path).exists()) {
            playConfirmSound()
            parentFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, LoadingFragment.newInstance(path))
                .addToBackStack(null)
                .commit()
        } else {
            Toast.makeText(requireContext(), "Image not found", Toast.LENGTH_SHORT).show()
        }
    }

    private fun playSwipeSound() {
        MediaPlayer.create(requireContext(), R.raw.pageturn).apply {
            setOnCompletionListener { it.release() }
            start()
        }
    }

    private fun playConfirmSound() {
        MediaPlayer.create(requireContext(), R.raw.sample_confirm_success).apply {
            setOnCompletionListener { it.release() }
            start()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        mediaPlayer?.release()
    }
}

class AccessibleImageView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : androidx.appcompat.widget.AppCompatImageView(context, attrs) {

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
