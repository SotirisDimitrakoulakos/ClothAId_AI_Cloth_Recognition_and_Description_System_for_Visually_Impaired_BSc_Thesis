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

    private fun capitalizeFirstLetter(text: String): String {
        return if (text.isNotEmpty()) text.substring(0, 1).uppercase() + text.substring(1) else text
    }

    private fun formatPredictions(jsonString: String): String {
        return try {
            val jsonObj = JSONObject(jsonString)
            val masterCategory = capitalizeFirstLetter(jsonObj.optString("masterCategory", "N/A"))
            val subCategory = capitalizeFirstLetter(jsonObj.optString("subCategory", "N/A"))
            var articleType = capitalizeFirstLetter(jsonObj.optString("articleType", "N/A"))
            val baseColour = capitalizeFirstLetter(jsonObj.optString("baseColour", "N/A"))
            val gender = capitalizeFirstLetter(jsonObj.optString("gender", "N/A"))
            val season = capitalizeFirstLetter(jsonObj.optString("season", "N/A"))
            val usage = capitalizeFirstLetter(jsonObj.optString("usage", "N/A"))
            if (articleType == "Tshirts"){
                articleType = "T-Shirts"
            }

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
