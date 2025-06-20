package com.example.clothaid_thesis

import android.os.Bundle
import android.view.*
import androidx.fragment.app.Fragment
import com.example.clothaid_thesis.databinding.FragmentLoadingBinding
import com.google.gson.Gson

class LoadingFragment : Fragment() {

    private var _binding: FragmentLoadingBinding? = null
    private val binding get() = _binding!!

    private var photoPath: String? = null

    companion object {
        private const val ARG_PHOTO_PATH = "photo_path"
        fun newInstance(photoPath: String) = LoadingFragment().apply {
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
        _binding = FragmentLoadingBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Start async network call to send photo and get results
        photoPath?.let { path ->
            NetworkUtils.uploadImageAndGetResult(requireContext(), path) { resultMap ->
                val jsonResult = Gson().toJson(resultMap)
                parentFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, ResultsFragment.newInstance(jsonResult, path))
                    .addToBackStack(null)
                    .commit()
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
