package com.example.clothaid_thesis

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var imageUri: Uri
    private lateinit var photoFile: File
    private lateinit var progressBar: ProgressBar
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var cameraButton: Button
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        progressBar = findViewById(R.id.progressBar)
        cameraButton = findViewById(R.id.cameraButton)

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.US
            }
        }

        cameraButton.setOnClickListener {
            dispatchTakePictureIntent()
        }
    }

    private fun dispatchTakePictureIntent() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = File.createTempFile("IMG_", ".jpg", getExternalFilesDir(Environment.DIRECTORY_PICTURES))
        imageUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri)
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            imageView.setImageURI(imageUri)
            uploadImageToServer(photoFile)
        }
    }

    private fun uploadImageToServer(imageFile: File) {
        progressBar.visibility = View.VISIBLE

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", "clothing.jpg",
                imageFile.asRequestBody("image/jpeg".toMediaType())
            )
            .build()

        val request = Request.Builder()
            .url("http://YOUR_LAPTOP_IP:5000/predict") // ⚠️ Replace with your local server IP
            .post(requestBody)
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "Upload failed: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body?.string()
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    responseBody?.let {
                        val result = parseResponse(it)
                        speakDescription(result)
                    }
                }
            }
        })
    }

    private fun parseResponse(json: String): String {
        val jsonObject = JSONObject(json)
        val description = StringBuilder()

        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = jsonObject.getString(key)
            description.append("$key: $value. ")
        }

        return description.toString()
    }

    private fun speakDescription(description: String) {
        textToSpeech.speak(description, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
    }
}
