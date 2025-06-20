package com.example.clothaid_thesis

import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.os.Bundle
import android.view.*
import androidx.fragment.app.Fragment
import com.example.clothaid_thesis.databinding.FragmentConfirmBinding

class ConfirmFragment : Fragment() {

    private var _binding: FragmentConfirmBinding? = null
    private val binding get() = _binding!!

    private var photoPath: String? = null
    private var mediaPlayer: MediaPlayer? = null

    companion object {
        private const val ARG_PHOTO_PATH = "photo_path"
        fun newInstance(photoPath: String) = ConfirmFragment().apply {
            arguments = Bundle().apply {
                putString(ARG_PHOTO_PATH, photoPath)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        photoPath = arguments?.getString(ARG_PHOTO_PATH)
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

        // Show the image
        photoPath?.let {
            val bitmap = BitmapFactory.decodeFile(it)
            binding.imageView.setImageBitmap(bitmap)
        }

        // Single tap: send photo to server (play ding sound)
        binding.root.setOnClickListener {
            mediaPlayer = MediaPlayer.create(requireContext(), R.raw.sample_confirm_success)
            mediaPlayer?.setOnCompletionListener {
                it.release()
                // Navigate after the sound completes
                parentFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, LoadingFragment.newInstance(photoPath!!))
                    .addToBackStack(null)
                    .commit()
            }
            mediaPlayer?.start()
        }

        // Long press: go back to camera (pop all fragments)
        binding.root.setOnLongClickListener {
            parentFragmentManager.popBackStack(null, androidx.fragment.app.FragmentManager.POP_BACK_STACK_INCLUSIVE)
            true
        }
    }

    private fun playConfirmSound() {
        mediaPlayer = MediaPlayer.create(requireContext(), R.raw.sample_confirm_success)
        mediaPlayer?.setOnCompletionListener {
            it.release()
        }
        mediaPlayer?.start()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        mediaPlayer?.release()
    }
}
