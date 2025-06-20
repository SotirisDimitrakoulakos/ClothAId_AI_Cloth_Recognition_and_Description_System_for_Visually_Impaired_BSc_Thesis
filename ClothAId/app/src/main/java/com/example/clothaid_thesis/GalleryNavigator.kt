package com.example.clothaid_thesis

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.MediaStore

object GalleryNavigator {
    fun getRecentImagePaths(context: Context, limit: Int = 50): List<String> {
        val imagePaths = mutableListOf<String>()
        val uri: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val projection = arrayOf(MediaStore.Images.Media.DATA)
        val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"

        val cursor: Cursor? = context.contentResolver.query(
            uri, projection, null, null, sortOrder
        )

        cursor?.use {
            val columnIndex = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
            while (it.moveToNext() && imagePaths.size < limit) {
                val imagePath = it.getString(columnIndex)
                imagePaths.add(imagePath)
            }
        }

        return imagePaths
    }
}
