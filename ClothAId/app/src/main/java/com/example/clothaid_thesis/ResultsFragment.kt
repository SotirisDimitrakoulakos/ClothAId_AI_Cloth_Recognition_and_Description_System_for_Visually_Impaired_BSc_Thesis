package com.example.clothaid_thesis

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.clothaid_thesis.databinding.FragmentResultsBinding
import org.json.JSONObject

class ResultsFragment : Fragment() {

    private var _binding: FragmentResultsBinding? = null
    private val binding get() = _binding!!

    companion object {
        private const val ARG_RESULT_TEXT = "result_text"
        private const val ARG_IMAGE_PATH = "image_path"

        fun newInstance(resultText: String, imagePath: String) = ResultsFragment().apply {
            arguments = Bundle().apply {
                putString(ARG_RESULT_TEXT, resultText)
                putString(ARG_IMAGE_PATH, imagePath)
            }
        }
    }

    private fun formatPredictions(jsonString: String): String {
        return try {
            val jsonObj = JSONObject(jsonString)
            val masterCategory = jsonObj.optString("masterCategory", "N/A")
            val subCategory = jsonObj.optString("subCategory", "N/A")
            val articleType = jsonObj.optString("articleType", "N/A")
            val baseColour = jsonObj.optString("baseColour", "N/A")
            val gender = jsonObj.optString("gender", "N/A")
            val season = jsonObj.optString("season", "N/A")
            val usage = jsonObj.optString("usage", "N/A")

            """
        Clothing Characteristics:
          Main Category: $masterCategory
          Sub-Category: $subCategory
          Precise Category: $articleType
          Base Color: $baseColour
          Gender: $gender
          Season: $season
          Usage: $usage
        """.trimIndent()
        } catch (e: Exception) {
            "Failed to parse results"
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        _binding = FragmentResultsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val resultJson  = arguments?.getString(ARG_RESULT_TEXT) ?: "{}"
        val imagePath = arguments?.getString(ARG_IMAGE_PATH)

        // Load and set the background image with some transparency
        if (!imagePath.isNullOrEmpty()) {
            val bitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            binding.imageBackground.setImageBitmap(bitmap)
            binding.imageBackground.alpha = 0.5f
        }

        val formattedResult = formatPredictions(resultJson)
        binding.textViewResults.text = formattedResult

        (activity as? MainActivity)?.speak(formattedResult)

        // Long press: go back to camera (pop all fragments)
        binding.invisibleLongPressOverlay.setOnLongClickListener {
            (activity as? MainActivity)?.stopSpeaking()
            parentFragmentManager.popBackStack(null, androidx.fragment.app.FragmentManager.POP_BACK_STACK_INCLUSIVE)
            true
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
