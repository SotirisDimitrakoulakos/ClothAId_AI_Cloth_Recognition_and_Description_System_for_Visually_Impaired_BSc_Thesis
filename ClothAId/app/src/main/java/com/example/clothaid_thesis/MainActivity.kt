package com.example.clothaid_thesis

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.commit
import java.util.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech
    private var isTtsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        loadPlaceholderFragment() // before requestPermissions()
        requestPermissions()
        tts = TextToSpeech(this, this)
    }

    private fun loadPlaceholderFragment() {
        supportFragmentManager.commit {
            replace(R.id.fragment_container, PlaceholderFragment())
        }
    }

    private fun requestPermissions() {
        val storagePermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        val requiredPermissions = arrayOf(
            Manifest.permission.CAMERA,
            storagePermission
        )

        val missing = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            loadCameraFragment()
        } else {
            val shouldExplain = missing.any {
                shouldShowRequestPermissionRationale(it)
            }

            if (shouldExplain) {
                // Show a custom dialog explaining why the permissions are needed.
                // After user agrees, call requestPermissionLauncher.launch(...)
            } else {
                requestPermissionLauncher.launch(missing.toTypedArray())
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            loadCameraFragment()
        } else {
            Toast.makeText(this, "Permissions denied. App may not work properly.", Toast.LENGTH_LONG).show()
        }
    }

    private fun loadCameraFragment() {
        supportFragmentManager.commit {
            replace(R.id.fragment_container, CameraFragment())
        }
    }

    override fun onInit(status: Int) {
        isTtsReady = status == TextToSpeech.SUCCESS
        if (isTtsReady) {
            tts.language = Locale("en", "US") // American English
        }
    }

    fun speak(text: String) {
        if (isTtsReady) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    fun stopSpeaking() {
        tts.stop()
    }

    override fun onDestroy() {
        tts.shutdown()
        super.onDestroy()
    }
}
