package com.example.clothaid_thesis

import android.content.Context
import okhttp3.*
import java.io.File
import java.io.IOException
import com.google.gson.Gson
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject

object NetworkUtils {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .build()
    private val gson = Gson()

    private const val SERVER_URL = "http://192.168.1.64:5000/predict"

    fun uploadImageAndGetResult(context: Context, imagePath: String, callback: (Map<String, String>) -> Unit) {
        val file = File(imagePath)
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", file.name,
                file.asRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url(SERVER_URL)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object: Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                Handler(Looper.getMainLooper()).post {
                    callback(emptyMap())
                    Toast.makeText(context, "Network failure: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.body?.string()?.let { bodyString ->
                    val jsonObject = JSONObject(bodyString)
                    val predictions = jsonObject.optJSONObject("predictions")
                    val map = mutableMapOf<String, String>()

                    predictions?.let {
                        val keys = it.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            map[key] = it.optString(key, "N/A")
                        }
                    }
                    Handler(Looper.getMainLooper()).post {
                        callback(map)
                    }
                } ?: run {
                    Handler(Looper.getMainLooper()).post {
                        callback(emptyMap())
                    }
                }
            }
        })
    }
}
